#!/bin/bash
# the path to the ComputeCPP package root directory, e.g. /home/user/ComputeCpp-CE-0.1-Linux/
PACKAGE_ROOT=$1

function install_gmock  {
  mkdir -p external && pushd external
  git clone git@github.com:google/googletest.git
  pushd googletest/googlemock/make && make
  popd 
  popd
}

function configure  {
  mkdir -p build && pushd build 
  cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$PACKAGE_ROOT  -DPARALLEL_STL_BENCHMARKS=ON
  popd
}

function mak  {
  pushd build && make -j32
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
