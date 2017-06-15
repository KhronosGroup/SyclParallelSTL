#!/bin/bash

VERSION=0.2.1

ln -sf /tmp/ComputeCpp-CE-${VERSION}-Linux/ /tmp/ComputeCpp-latest

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/CL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/CL/sycl_builtins.h.h

cat .travis/additional_undef /tmp/ComputeCpp-latest/include/CL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-latest/include/CL/host_relational_builtins.h

