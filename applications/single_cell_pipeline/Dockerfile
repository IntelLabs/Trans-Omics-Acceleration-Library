# MIT License

# Copyright (c) 2022 Intel Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Authors: Narendra Chaudhary <narendra.chaudhary@intel.com>; Sanchit Misra <sanchit.misra@intel.com>

# Install Base miniconda image
ARG BASE_IMAGE=continuumio/miniconda3
FROM ${BASE_IMAGE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git build-essential gcc curl gnupg gnupg2 gnupg1

RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common intel-basekit intel-hpckit && \
  rm -rf /var/lib/apt/lists/*


RUN conda install python==3.8.0
RUN conda install -y seaborn scikit-learn=1.0.2 statsmodels numba pytables
RUN conda install -y -c conda-forge mkl-service
RUN conda install -y -c conda-forge python-igraph leidenalg
RUN conda install -y -c conda-forge cython jinja2 clang-tools
RUN conda install -y -c katanagraph/label/dev -c conda-forge katana-python


RUN pip install scanpy==1.8.1
#RUN pip install scikit-learn-intelex

#################  Using the compiled oneDAL code ###################################
RUN git clone https://github.com/oneapi-src/oneDAL.git
WORKDIR /oneDAL
COPY ./tsne_gradient_descent_fpt.cpp /oneDAL/cpp/daal/src/algorithms/tsne/

RUN ./dev/download_micromkl.sh
RUN . /opt/intel/oneapi/compiler/latest/env/vars.sh intel64 && \
    . /opt/intel/oneapi/tbb/latest/env/vars.sh intel64 && \
    . /opt/intel/oneapi/dpl/latest/env/vars.sh intel64 && \
    make daal PLAT=lnx32e COMPILER=icc CORE.ALGORITHMS.CUSTOM=tsne -j && \
    make _daal _release_c PLAT=lnx32e COMPILER=icc -j

WORKDIR /
RUN git clone https://github.com/intel/scikit-learn-intelex.git

ENV NO_DIST=1
ENV OFF_ONEDAL_IFACE=1

WORKDIR /scikit-learn-intelex

RUN . /oneDAL/__release_lnx/daal/latest/env/vars.sh intel64 \
    && python setup.py install --single-version-externally-managed --record=record.txt \
    && python setup_sklearnex.py install --single-version-externally-managed --record=record_sklearnex.txt

RUN pip install pybind11
RUN pip install jupyterlab
RUN pip install wget


WORKDIR /workspace
ENV HOME /workspace
COPY ./notebooks/ /workspace


RUN pip uninstall -y umap-learn
WORKDIR /
RUN git clone --recursive https://github.com/IntelLabs/Trans-Omics-Acceleration-Library.git
WORKDIR Trans-Omics-Acceleration-Library/applications/UMAP_fast/umap_extend/
RUN python setup.py install

WORKDIR ../umap/
RUN python setup.py install  

# ENV NUMEXPR_MAX_THREADS=64
# ENV NUMBA_NUM_THREADS=64

WORKDIR /workspace
CMD . /oneDAL/__release_lnx/daal/latest/env/vars.sh intel64 \
    && . /opt/intel/oneapi/tbb/latest/env/vars.sh intel64 \
	 && jupyter notebook \
		--no-browser \
		--allow-root \
		--port=8888 \
		--ip=0.0.0.0 \
		--notebook-dir=/workspace \
		--NotebookApp.password="" \
		--NotebookApp.token="" \
		--NotebookApp.password_required=False

# build with "docker build -t scanpy ."
# run with "docker run -it -p 8888:8888 -v ~/Trans-Omics-Acceleration-Library/applications/single_cell_pipeline/data:/data scanpy"
