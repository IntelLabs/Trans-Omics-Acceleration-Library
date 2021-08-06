# Instructions for
# ubuntu

# Update
apt-get update

# Install software
apt-get install -y git
apt-get install -y libcurl4-openssl-dev
apt-get install -y hdf5-tools
apt-get install -y rsync
apt-get install -y make
apt-get install -y gcc
apt-get install -y libblas-dev
apt-get install -y python3.7 python3-pip
ln -nsf /usr/bin/python3.7 /usr/bin/python

conda create --name Atac python=3.7
conda activate Atac

cd /home/
# Clone the libxsmm repository and set library path
git clone https://github.com/hfp/libxsmm.git
cd /home/libxsmm && make -j AVX=3 && cd -               # Use AVX=2 for AVX2
export LD_LIBRARY_PATH=/home/libxsmm/lib/


# Clone atacworks repo
git clone --branch v0.2.0 https://github.com/clara-parabricks/AtacWorks.git

cd  /home/AtacWorks/

# Copy patch to AtacWorks folder and Apply patch
git apply AtacWorks_cpu_optimization_patch.patch

python3.7 -m pip install -r requirements-base.txt
python3.7 -m pip install torch torchvision torchaudio
python3.7 -m pip install -r requirements-macs2.txt

# # Install torch-ccl
# git clone --branch v1.1.0 https://github.com/intel/torch-ccl.git && cd torch-ccl
# git submodule sync
# git submodule update --init --recursive
# python3.7 setup.py install

# Setup 1D convolution module
cd /home/libxsmm/samples/deeplearning/conv1dopti_layer/Conv1dOpti-extension/ && python setup.py install && cd -
python3.7 -m pip install .


# Set path
atacworks=/home/AtacWorks/

# Download test data
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/noisy_data/dsc.1.Mono.50.cutsites.smoothed.200.bw
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.bw
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak

rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /home/

rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /home/
export PATH="$PATH:/home/" >> /home/.bashrc

# This command reads the peak positions from the .narrowPeak file and writes them to a bigWig file in the current directory,
# named `dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw`.
python $atacworks/scripts/peak2bw.py \
    --input dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak \
    --sizes $atacworks/data/reference/hg19.chrom.sizes \
    --out_dir ./ \
    --skip 1

python $atacworks/scripts/get_intervals.py \
    --sizes $atacworks/data/reference/hg19.auto.sizes \
    --intervalsize 50000 \
    --out_dir ./ \
    --val chr20 \
    --holdout chr10

python $atacworks/scripts/bw2h5.py \
        --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
        --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
        --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
        --intervals training_intervals.bed \
        --out_dir ./ \
        --prefix Mono.50.2400.train \
        --pad 5000 \
        --nonzero

python $atacworks/scripts/bw2h5.py \
        --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
        --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
        --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
        --intervals val_intervals.bed \
        --out_dir ./ \
        --prefix Mono.50.2400.val \
        --pad 5000


export KMP_AFFINITY=compact,1,0,granularity=fine
export OMP_NUM_THREADS=27                           # (Available cores - 1)

numactl --membind 0 -C 1-27 python $atacworks/scripts/main.py train \
        --config configs/train_config.yaml \
        --config_mparams configs/model_structure.yaml \
        --files_train $atacworks/Mono.50.2400.train.h5 \
        --val_files $atacworks/Mono.50.2400.val.h5

# Another option to use on machines without NUMA --- "taskset -c 1-3 python "