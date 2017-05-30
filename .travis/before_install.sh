#!/bin/bash

set -ev

# Add modern dependencies (cmake, boost)
sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
sudo add-apt-repository ppa:kzemek/boost -y
sudo apt-get update -q
# TriSYCL requires modern boost
# Use the system OpenCL loader
sudo apt-get install cmake boost1.58 ocl-icd-libopencl1 ocl-icd-dev opencl-headers -y
# Use gcc 5 as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
