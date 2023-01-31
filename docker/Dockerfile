ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.2.0
ARG CUDA=11.2
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}
LABEL Description="Machine Learning Docker"

ARG CUDA_VERSION
ARG CUDA
ARG UID=
ARG USER_NAME=
ARG PYTHON_VERSION=3.8

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND noninteractive 

# Install packages
RUN apt-get update && apt-get install -y \
    locales \
    ca-certificates \
    cmake \
    lsb-release \
    mesa-utils \
    git \
    subversion \
    nano \
    terminator \
    xterm \
    wget \
    curl \
    htop \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    gdb valgrind && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# For CUDA profiling
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig


# Install tmux 3.2
RUN apt-get update && apt-get install -y automake autoconf pkg-config libevent-dev libncurses5-dev bison && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tmux/tmux.git && \
    cd tmux && git checkout tags/3.2 && ls -la && sh autogen.sh && ./configure && make -j8 && make install


# Install new paramiko (solves ssh issues)
RUN apt-add-repository universe
RUN apt-get update && apt-get install -y python3-pip python3 build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN /usr/bin/yes | pip3 install --upgrade pip
RUN /usr/bin/yes | pip3 install --upgrade virtualenv
RUN /usr/bin/yes | pip3 install --upgrade paramiko
RUN /usr/bin/yes | pip3 install --ignore-installed --upgrade numpy protobuf

# Locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# requirements (optinonal)
# COPY requirements.txt requirements.txt
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip install -r requirements.txt
# COPY . .

# installing pytorch (optional)
# RUN /usr/bin/yes | pip3 install -f torch torchvision
RUN pip3 install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113

# installing required packages (optional)
RUN python3 -m pip install pandas
RUN python3 -m pip install matplotlib
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install seaborn
RUN python3 -m pip install opencv-python

# installing tensorflow 2.9.1
RUN pip3 install tensorflow==2.9.1

# installing transformers
RUN pip3 install transformers

# Jupyter lab
RUN python3 -m pip install jupyterlab

# Terminator Config (optional)
RUN mkdir -p /root/.config/terminator/
COPY terminator_config /root/.config/terminator/config 

WORKDIR /home

# Launch terminator
CMD ["terminator"]