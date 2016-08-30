#!/bin/bash
# the path to the sycl sdk, e.g. ~/computecpp-sdk/
SDK_PATH=$1
# the path to the ComputeCPP package root directory, e.g. ~/ComputeCPP-16.07-Linux/
PACKAGE_ROOT=$2

function install_gmock  {
  mkdir -p external && pushd external
  pushd googletest/googlemock/make && make
  popd 
  popd
}

function configure  {
  mkdir -p build && pushd build 
  cmake .. -DCMAKE_MODULE_PATH=$SDK_PATH/cmake/Modules/ -DCOMPUTECPP_PACKAGE_ROOT_DIR=$PACKAGE_ROOT -DSYCL_PATH=$PACKAGE_ROOT -DOpenCL_INCLUDE_DIR=$PACKAGE_ROOT/include/CL/
  popd
}

function mak  {
  pushd build && make 
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
