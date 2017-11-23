#!/bin/bash

VERSION=0.4.0

ln -sf /tmp/ComputeCpp-CE-${VERSION}-Ubuntu-16.04-64bit/ /tmp/ComputeCpp-latest

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/SYCL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/SYCL/sycl_builtins.h.h

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/SYCL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/SYCL/host_relational_builtins.h

