#!/bin/bash


cat .travis/additional_undef /tmp/ComputeCpp-CE-0.2.0-Linux/include/CL/sycl_builtins.h.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.2.0-Linux/include/CL/sycl_builtins.h

cat .travis/additional_undef /tmp/ComputeCpp-CE-0.2.0-Linux/include/CL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.2.0-Linux/include/CL/host_relational_builtins.h

