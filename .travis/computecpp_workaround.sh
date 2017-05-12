#!/bin/bash

VERSION=0.2.0
cat .travis/additional_undef /tmp/ComputeCpp-CE-${VERSION}-Linux/include/CL/sycl_builtins.h.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-${VERSION}-Linux/include/CL/sycl_builtins.h.h

cat .travis/additional_undef /tmp/ComputeCpp-CE-${VERSION}-Linux/include/CL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-${VERSION}-Linux/include/CL/host_relational_builtins.h
