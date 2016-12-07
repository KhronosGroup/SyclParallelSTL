#!/bin/bash
# the path to the ComputeCPP package root directory, e.g. /home/user/ComputeCpp-CE-0.1-Linux/
PACKAGE_ROOT=$1

#compute the number of cores
NPROC=$(nproc)

function install_gmock  {(
  REPO="git@github.com:google/googletest.git"
  #REPO="https://github.com/google/googletest.git"
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
  cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$PACKAGE_ROOT  -DPARALLEL_STL_BENCHMARKS=ON
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
