#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget https://computecpp.codeplay.com/downloads/computecpp-ce/latest/ubuntu-14.04-64bit.tar.gz
rm -rf /tmp/ComputeCpp-latest && mkdir /tmp/ComputeCpp-latest/
tar -xzf ubuntu-14.04-64bit.tar.gz -C /tmp/ComputeCpp-latest --strip-components 1
ls -R /tmp/ComputeCpp-latest/
# Workaround for C99 definition conflict
bash .travis/computecpp_workaround.sh
