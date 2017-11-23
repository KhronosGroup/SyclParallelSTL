#!/bin/bash

set -ev

# Add modern dependencies (cmake, boost)
sudo add-apt-repository ppa:kzemek/boost -y
sudo apt-get update -q
# TriSYCL requires modern boost
# Use the system OpenCL loader
sudo apt-get install boost1.58 libboost-chrono1.58-dev libboost-log1.58-dev ocl-icd-libopencl1 ocl-icd-dev opencl-headers -y

# triSYCL requires newer cmake than in apt
wget https://cmake.org/files/v3.5/cmake-3.5.1-Linux-x86_64.tar.gz -O /tmp/cmake.tar.gz
tar -xzf /tmp/cmake.tar.gz -C /tmp
sudo cp -rf /tmp/cmake-3.5.1-Linux-x86_64/* /usr/
