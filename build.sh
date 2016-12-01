#!/bin/bash
# the path to the ComputeCPP package root directory, e.g. /home/user/ComputeCpp-CE-0.1-Linux/

# Useless to go on when an error occurs
set -o errexit

# How to use build.sh to compile SyclParallelSTL with ComputeCpp ?
# ./build.sh "path/to/ComputeCpp"
#
# How to use build.sh to compile SyclParallelSTL with triSYCL ?
# ./build.sh --trisycl [-DTRISYCL_INCLUDE_DIR=path/to/triSYCL/include] [-DBOOST_COMPUTE_INCLUDE_DIR=path/to/boost/compute/include]
#
# paths can be relative ones, this script will make them absoulte before to send them to CMake
#
# WARNING/TODO: setting the "path/to/compute" in triSYCL mode is not an option yet
#

if [ $1 == "--trisycl" ]
then
	shift
	echo "build.sh enter mode: triSYCL"
	CMAKE_ARGS="$CMAKE_ARGS -DUSE_COMPUTECPP=OFF"
else
	echo "build.sh enter mode: ComputeCpp"
	CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
	shift
fi
NPROC=$(nproc)

function install_gmock  {(
  #REPO="git@github.com:google/googletest.git"
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
  if [ $@ ]
  then
	  echo "cmake additionnal args: $@"
  fi
  cmake .. $CMAKE_ARGS -DPARALLEL_STL_BENCHMARKS=ON $@
  popd
}

function mak  {
  pushd build && make -j$NPROC
  popd
}

function tst {
  pushd build/tests
  ctest
  popd 
}

function main {
  install_gmock
  configure
  mak
  tst
}

main
