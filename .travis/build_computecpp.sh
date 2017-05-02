#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget https://computecpp.codeplay.com/downloads/computecpp-ce/latest/ubuntu-14.04-64bit.tar.gz
tar -xzf ubuntu-14.04-64bit.tar.gz -C /tmp 
# Workaround for C99 definition conflict
bash .travis/computecpp_workaround.sh
