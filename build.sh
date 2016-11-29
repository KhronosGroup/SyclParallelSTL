#!/bin/bash
# the path to the ComputeCPP package root directory, e.g. /home/user/ComputeCpp-CE-0.1-Linux/
PACKAGE_ROOT=$1

function install_gmock  {(
  REPO="git@github.com:google/googletest.git"
  #REPO="https://github.com/google/googletest.git"
  if [ -d external ]
  then
    cd external
    if [ -d googletest ]
    then
      cd googletest
      git pull
    else
      git clone $REPO
      cd googletest
    fi
  else
    mkdir external
    cd external
    git clone $REPO
    cd googletest
  fi
  cd googlemock/make
  make -j$NPROC
)}

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
