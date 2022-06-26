
# Download entire repository
```bash
git clone --recursive https://github.com/IntelLabs/Trans-Omics-Acceleration-Library.git
```
# Create an Anaconda environment
```bash
conda create --name single_cell python=3.8.0
conda activate single_cell
```

## Necessary scanpy tools
```bash
conda install -y seaborn scikit-learn statsmodels numba pytables
conda install -y -c conda-forge mkl-service
conda install -y -c conda-forge python-igraph leidenalg
conda install -y -c conda-forge cython jinja2 clang-tools
conda install -y -c katanagraph/label/dev -c conda-forge katana-python==0.1.102.dev+108.0.0d2943
```

## Install scanpy
```bash
pip install scanpy==1.8.1
```

## (Slower version) Install scikit-learn intel extension (PIP version)
```bash
pip install scikit-learn-intelex
```

## Preferably compile and install daal, daal4py, scikit-learn-intelex (For faster tSNE)
```bash
git clone https://github.com/oneapi-src/oneDAL.git            # Download daal
cd ./oneDAL
cp ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/tsne_gradient_descent_fpt.cpp cpp/daal/src/algorithms/tsne/            # Replace the tsne algorithm

source /opt/intel/oneapi/compiler/latest/env/vars.sh intel64
source /opt/intel/oneapi/tbb/latest/env/vars.sh intel64
source /opt/intel/oneapi/dpl/latest/env/vars.sh intel64

./dev/download_micromkl.sh                                    #Download microMKL libs via script 

make daal PLAT=lnx32e COMPILER=icc CORE.ALGORITHMS.CUSTOM=tsne -j64      # Use "make clean PLAT=lnx32e" for cleaning before this step
make _daal _release_c PLAT=lnx32e COMPILER=icc -j64           # compile entire daal with ICC
export NO_DIST=1
export OFF_ONEDAL_IFACE=1
source ./__release_lnx/daal/latest/env/vars.sh intel64        # source the compiled daal
cd ..

git clone https://github.com/intel/scikit-learn-intelex.git
cd scikit-learn-intelex/

python setup.py install --single-version-externally-managed --record=record.txt       # Use "git clean -dfx" for cleaning before this step
python setup_sklearnex.py install --single-version-externally-managed --record=record_sklearnex.txt
```

## Install other packages
```bash
pip install pybind11
```

## Install umap_extend and umap 
```bash

pip uninstall umap-learn
cd ~/Trans-Omics-Acceleration-Library/applications/UMAP_fast/umap_extend
python setup.py install                          # Uncomment AVX512 lines in setup.py before doing this step on avx512 machines


cd ~/Trans-Omics-Acceleration-Library/applications/UMAP_fast/umap
python setup.py develop                     # do python setup.py install if moving environment using conda-pack
```


## Example Dataset
The dataset was made publicly available by 10X Genomics. Use the following command to download the count matrix for this dataset and store it in the data folder:
```bash
wget -P ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad
```

## Setup and run
```bash
export NUMEXPR_MAX_THREADS=56          # equal to number of threads on a single socket
export NUMBA_NUM_THREADS=56            # Remember to delete __pycache__ folder from local directory and umap/umap/ directory if increasing number of threads

# also update sc.settings.n_jobs=56 to set number of threads inside 1M_brain_cpu_analysis.py

cd ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/notebooks/

python -m sklearnex 1M_brain_cpu_analysis.py

# Or the jupyter notebook with sklearn patch in it. 
# from sklearnex import patch_sklearn
# patch_sklearn()

#jupyter notebook
```

## Docker instructions
```bash
cd ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/
docker build -t scanpy .           # Create a docker image named scanpy

# Download dataset
wget -P ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad

docker run -it -p 8888:8888 -v ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/data:/data scanpy   # run docker container with the data folder as volume
```