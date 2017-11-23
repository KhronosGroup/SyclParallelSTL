#!/bin/bash

set -ev

# Add modern dependencies (cmake, boost)
sudo add-apt-repository ppa:kzemek/boost -y
sudo apt-get update -q
# TriSYCL requires modern boost
# Use the system OpenCL loader
sudo apt-get install boost1.58 libboost-chrono1.58-dev libboost-log1.58-dev ocl-icd-libopencl1 ocl-icd-dev opencl-headers -y

# CMake 3.5 for triSYCL
wget https://cmake.org/files/v3.5/cmake-3.5.1-Linux-x86_64.tar.gz -O /tmp/cmake.tar.gz
tar -xzf /tmp/cmake.tar.gz -C /tmp
sudo cp -rf /tmp/cmake-3.5.1-Linux-x86_64/* /usr/

# Boost.Compute for triSYCL
wget https://github.com/boostorg/compute/archive/master.zip -O /tmp/boost.compute.zip
unzip /tmp/boost.compute.zip -d /tmp &> /dev/null
sudo cp -rf /tmp/compute-master/include /usr/

# Clang 4.0 release with OpenMP for trySYCL
wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz -O /tmp/clang-4.0.tar.xz
tar -xJf /tmp/clang-4.0.tar.xz -C /tmp
sudo cp -rf /tmp/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-14.04/* /usr/lib/llvm-4.0/
