#!/bin/bash
# the path to the ComputeCPP package root directory, e.g. /home/user/ComputeCpp-CE-0.1-Linux/

# How to use build.sh to compile SyclParallelSTL with ComputeCpp ?
# ./build.sh "path/to/ComputeCpp"
#
# How to use build.sh to compile SyclParallelSTL with triSYCL ?
# ./build.sh --trisycl "path/to/triSYCL" ["path/to/compute"]
#
# paths can be relative ones, this script will make them absoulte before to send them to CMake
#
# 

if [ $1 == "--trisycl" ]
then
	echo "build.sh enter mode: triSYCL"
	CMAKE_ARGS="$CMAKE_ARGS -DUSE_COMPUTECPP=OFF -DTRISYCL_PACKAGE_ROOT_DIR=$(readlink -f $2)"
	if [ $3 ]
	then
		CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTE_PACKAGE_ROOT_DIR=$(readlink -f $3)"
	fi
else
	echo "build.sh enter mode: ComputeCpp"
	CMAKE_ARGS="$CMAKE_ARGS -DUSE_COMPUTECPP=ON -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
fi
NPROC=$(nproc)

function install_gmock  {
  mkdir -p external && pushd external
  git clone git@github.com:google/googletest.git
  pushd googletest/googlemock/make && make
  popd 
  popd
}

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
