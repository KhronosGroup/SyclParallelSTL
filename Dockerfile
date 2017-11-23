FROM ubuntu:xenial

# Default values for the build
ARG git_branch
ARG git_slug
ARG c_compiler
ARG cxx_compiler
ARG impl

RUN apt-get -y update

# Utilities
RUN apt-get install -y --allow-downgrades --allow-remove-essential             \
    --allow-change-held-packages git wget apt-utils cmake unzip                \
    libboost-all-dev software-properties-common python-software-properties libcompute-dev

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -y update

# Clang 4.0
RUN if [ "${c_compiler}" = 'clang-4.0' ]; then apt-get install -y              \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
     clang-4.0 libomp-dev; fi

# GCC 6
RUN if [ "${c_compiler}" = 'gcc-6' ]; then apt-get install -y                  \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
    g++-6 gcc-6; fi

# OpenCL
RUN apt-get install -y --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /SyclParallelSTL

# SYCL
RUN if [ "${impl}" = 'triSYCL' ]; then cd /SyclParallelSTL && bash /SyclParallelSTL/.travis/build_triSYCL.sh; fi
RUN if [ "${impl}" = 'COMPUTECPP' ]; then cd /SyclParallelSTL && bash /SyclParallelSTL/.travis/build_computecpp.sh; fi

ENV CC=${c_compiler}
ENV CXX=${cxx_compiler}
ENV SYCL_IMPL=${impl}

CMD cd /SyclParallelSTL && \
    if [ "${SYCL_IMPL}" = 'triSYCL' ]; then \
      ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=/tmp/triSYCL-master/include; \
    elif [ "${SYCL_IMPL}" = 'COMPUTECPP' ]; then \
      COMPUTECPP_TARGET="host" ./build.sh /tmp/ComputeCpp-latest; \
    else \
      echo "Unknown SYCL implementation ${SYCL_IMPL}"; return 1; \
    fi
