#!/bin/bash

# How to use build.sh to compile SyclParallelSTL with ComputeCpp ?
# ./build.sh "path/to/ComputeCpp" (this path can be relative)
#
# for example:
#  ./build.sh /home/user/ComputeCpp
#
# How to use build.sh to compile SyclParallelSTL with triSYCL ?
# ./build.sh --trisycl [-DTRISYCL_INCLUDE_DIR=path/to/triSYCL/include] [-DBOOST_COMPUTE_INCLUDE_DIR=path/to/boost/compute/include]
#
# for example (Ubuntu 16.04):
#  ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=~/triSYCL/include -DBOOST_COMPUTE_INCLUDE_DIR=~/compute/include
#
#


# Useless to go on when an error occurs
set -o errexit

if [ $1 == "--trisycl" ]
then
  shift
  echo "build.sh enter mode: triSYCL"
  CMAKE_ARGS="$CMAKE_ARGS -DUSE_COMPUTECPP=OFF $@"
else
  echo "build.sh enter mode: ComputeCpp"
  CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
  shift
fi
NPROC=$(nproc)

function install_gmock  {(
  REPO="https://github.com/google/googletest.git"
  mkdir -p external
  cd external
  if [ -d googletest ]
  then
    cd googletest
    git pull
  else
    git clone $REPO
    cd googletest
  fi
  cd googlemock/make
  make -j$NPROC
)}

function configure  {
  mkdir -p build && pushd build
  cmake .. $CMAKE_ARGS -DPARALLEL_STL_BENCHMARKS=ON 
  popd
}

function mak  {
  pushd build && make -j$NPROC
  popd
}

function tst {
  pushd build/tests
  ctest -j$NPROC
  popd
}

function main {
  install_gmock
  configure
  mak
  tst
}

main
