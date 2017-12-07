#!/bin/bash

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/SYCL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/SYCL/sycl_builtins.h

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/SYCL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/SYCL/host_relational_builtins.h
