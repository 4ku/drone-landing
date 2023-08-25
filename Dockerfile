# Use an official NVIDIA image with CUDA 11.8 and Python 3.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and Python 3.8
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-venv \
    curl \
    git \
    libgl1-mesa-glx

# Update pip
RUN pip3 install --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Copy your project files into the Docker image
COPY . /app/

RUN /bin/bash -c "pip install setuptools==57.1.0 && \
    pip install -e /app/ && \
    pip install -r /app/requirements.txt"

# Set the environment variables as per your setup
ENV CUDNN_PATH /usr/local/cuda/lib64
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$CUDNN_PATH:$LD_LIBRARY_PATH
WORKDIR /app
