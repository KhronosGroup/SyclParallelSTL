#!/bin/bash

cat .travis/additional_undef /tmp/ComputeCpp-CE-0.2.0-Linux/include/SYCL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.2.0-Linux/include/SYCL/sycl_builtins.h

cat .travis/additional_undef /tmp/ComputeCpp-CE-0.2.0-Linux/include/SYCL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.2.0-Linux/include/SYCL/host_relational_builtins.h
