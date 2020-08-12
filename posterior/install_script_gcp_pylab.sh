#!/bin/bash 

# Tested on Debian GNU/Linux 9 (stretch)
# Install the bare essentials
sudo apt-get install git bzip2 libxml2-dev
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils

# Install Miniconda and activate the base
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source .bashrc

# Clone repositories of PyLab and MDSINE2
git clone https://github.com/gerberlab/PyLab.git
git clone https://github.com/gerberlab/MDSINE2.git

# Create environment and install pylab
conda create -n pylab-env python=3.7.3
conda activate pylab-env
cd PyLab
pip install .
cd ..