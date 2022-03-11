# -*- mode: dockerfile -*-

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        build-essential \
        cmake \
        curl \
        git \
        ninja-build \
        wget

WORKDIR /opt/conda_setup

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x miniconda.sh && \
     ./miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip

RUN conda install pytorch numpy cudatoolkit=11.3 -c pytorch

WORKDIR /opt/moolib

COPY . /opt/moolib/

RUN pip install -r examples/requirements.txt

RUN pip install '.[all]'

WORKDIR /opt/moolib

CMD ["bash", "-c", "python examples/a2c.py"]

# Docker commands:
#   docker rm moolib -v
#   docker build -t moolib -f Dockerfile .
#   docker run --gpus all --rm --name moolib moolib
# or
#   docker run --gpus all -it --entrypoint /bin/bash moolib