#!/bin/bash

set -ev

###########################
# Get Intel OpenCL Runtime
###########################

wget http://registrationcenter-download.intel.com/akdlm/irc_nas/12513/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz -O /tmp/opencl_runtime.tgz
tar -xzf /tmp/opencl_runtime.tgz -C /tmp
sed 's/decline/accept/g' -i /tmp/opencl_runtime_16.1.2_x64_rh_6.4.0.37/silent.cfg
apt-get install -yq cpio
/tmp/opencl_runtime_16.1.2_x64_rh_6.4.0.37/install.sh -s /tmp/opencl_runtime_16.1.2_x64_rh_6.4.0.37/silent.cfg
