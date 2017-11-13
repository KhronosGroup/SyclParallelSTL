SYCL Parallel STL [![Build Status](https://travis-ci.org/KhronosGroup/SyclParallelSTL.svg?branch=master)](https://travis-ci.org/KhronosGroup/SyclParallelSTL)
==============================

This project features an implementation of the Parallel STL library
using the Khronos SYCL standard.

What is Parallel STL
-----------------------

Parallel STL is an implementation of the Technical Specification for C++
Extensions for Parallelism, current document number
[N4507](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4507.pdf).
This technical specification describes _a set of requirements for
implementations of an interface that computer programs written in the
C++ programming language may use to invoke algorithms with parallel
execution_.
In practice, this specification is aimed at the next C++ standard (C++ 17) and
offers the opportunity to users to specify _execution policies_ to
traditional STL algorithms which will enable the execution of
those algorithms in parallel.
The various policies can specify different kinds of parallel execution.
For example,

    std::vector<int> v = ...
    // Traditional sequential sort
    std::sort(vec.begin(), vec.end());
    // Explicit sequential sort
    std::sort(seq, vec.begin(), vec.end());
    // Explicit parallel sort
    std::sort(par, vec.begin(), vec.end());


What is SYCL?
----------------------

[SYCL](https://www.khronos.org/opencl/sycl) is a royalty-free,
cross-platform C++ abstraction layer that builds on top of OpenCL.
SYCL enables single-source development of OpenCL applications in C++ whilst
enabling traditional host compilers to produce standard C++ code.

SyclParallelSTL
---------------------

SyclParallelSTL exposes a SYCL policy in the experimental::parallel namespace
that can be passed to standard STL algorithms for them to run on SYCL.
Currently, only some STL algorithms are implemented, such as:

* sort : Bitonic sort for ranges where the size is a power of two, or sequential
  sort otherwise.
* transform : Parallel iteration (one thread per element) on the device.
* for\_each  : Parallel iteration (one thread per element) on the device.
* for\_each\_n : Parallel iteration (one work-item per element) on the device.
* count\_if : Parallel iteration (one work-item per 2 elements) on device.
* reduce : Parallel iteration (one work-item per 2 elements) on device.
* inner\_product: Parallel iteration (one work-item per 2 elements) on device.
* transform\_reduce : Parallel iteration (one work-item per 2 elements) on device.

Some optimizations are implemented. For example:

* the ability to pass iterators to buffers rather than STL containers to reduce
the amount of information copied in and out
* the ability to specify a queue to the SYCL policy so that the queue is used
for the various kernels (potentially enabling asynchronous execution of the calls).

## Building the project

This project currently supports the SYCL beta implementation from Codeplay,
ComputeCpp and the open-source triSYCL implementation.

The project uses CMake 3.5 in order to produce build files,
but more recent versions may work.

### Linux

In Linux, simply create a build directory and run CMake as follows:

    $ mkdir build
    $ cd build
    $ cmake ../ -DCOMPUTECPP_PACKAGE_ROOT_DIR=/path/to/sycl \
    $ make

Usual CMake options are available (e.g. building debug or release).
Makefile and Ninja generators are supported on Linux.

To simplify configuration, the `FindComputeCpp` cmake module from the ComputeCpp
SDK is included verbatim in this package within the `cmake/Modules/` directory.

If Google Mock is found in external/gmock, a set of unit tests is built.
Unit tests can be run by running Ctest in the binary directory. To install
gmock, run the following commands from the root directory of the SYCL parallel
stl project:

    $ mkdir external
    $ cd external
    $ git clone git@github.com:google/googletest.git
    $ cd googletest/googlemock/make
    $ make

To enable building the benchmarks, enable the *PARALLEL_STL_BENCHMARKS* option
in the cmake configuration line, i.e. `-DPARALLEL_STL_BENCHMARKS=ON`.

When building with a SYCL implementation that has no device compiler,
enable the *SYCL_NO_DEVICE_COMPILER* option to disable the specific
CMake rules for intermediate file generation.

Refer to your SYCL implementation documentation for
implementation-specific building options.

To quickly build the project and run some non-regression tests with
benchmarks, you can use the script `build.sh`:

If you want to compile it with ComputeCpp:

    ./build.sh "path/to/ComputeCpp" (this path can be relative)

for example (on Ubuntu 16.04):

    ./build.sh ~/ComputeCpp

If you want to compile it with triSYCL:

    ./build.sh --trisycl [-DTRISYCL_INCLUDE_DIR=path/to/triSYCL/include] [-DBOOST_COMPUTE_INCLUDE_DIR=path/to/boost/compute/include] [-DTRISYCL_OPENCL=ON]
for example (on Ubuntu 16.04):

    ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=~/triSYCL/include -DBOOST_COMPUTE_INCLUDE_DIR=~/compute/include [-DTRISYCL_OPENCL=ON]

or if Boost compute is in your library's default path, just with:

    ./build.sh --trisycl -DTRISYCL_INCLUDE_DIR=~/triSYCL/include [-DTRISYCL_OPENCL=ON]


Just run `build.sh` alone to get a small help message.

For triSYCL some benchmarks may display messages saying that unimplemented
features are used, you can ignore those messages as these features do not affect
the benchmarks executions, if you wish you can also contribute to the triSYCL
implementation to make those messages definitely disapear.

### Microsoft Windows

**Note: this is still under development, and has only been tested on Microsoft Windows 10.**

**Note: these steps assume the user will be using Microsfot Visual Studio 2017. Subtle details may
vary if you are using Microsoft Visual Studio 2015, or a version of Visual Studio released _after_
2017. The supported compiler is Microsoft Visual C++ 2015, not Microsoft Visual C++ 2017.**

1. Download and install the Microsoft Visual C++ 2015 compiler (typically achieved by installing
   [Microsoft Visual Studio 2017 Community Edition](https://www.visualstudio.com/downloads)). When
   installing the compiler, be sure to install the relevant SDK for your operating system (for
   example, on Windows 8.1, you will install the Windows 8.1 SDK; on Windows 10, you install the
   latest Windows 10 SDK).
2. Download and install the [Codeplay ComputeCpp Community Edition SDK for Windows]().
3. Open either _VS2015 x64 Native Tools Command Prompt_ or _Developer Command Prompt for VS 2017_.
4. Enter the following commands
```bash
mkdir SyclParallelSTL
cd SyclParallelSTL
mkdir build-debug
git clone https://github.com/KhronosGroup/SyclParallelSTL.git
cd build-debug
cmake -G"Visual Studio 14 2015 Win64" -DCOMPUTECPP_PACKAGE_ROOT_DIR=<path-to-ComputeCpp> ../SyclParallelSTL
```

where `<path-to-ComputeCpp>` is the path to your ComputeCpp directory. For example, a default
installation of ComputeCpp on a 64-bit build of Windows 10 will be `C:\\Program Files\\Codeplay\\ComputeCpp`.

5. Open `SyclSTL.sln` using Microsoft Visual Studio. If you are not using Visual Studio 2015, be 
   sure not to convert the solution to a later version.
6. Right-mouse click on `ALL_BUILD (Visual Studio 2015)` and select `Build`.

### Microsoft Windows Subsystem for Linux (Bash on Windows)

#### Prerequisites

* Ensure that WSL has been installed after installing the Windows 10 Fall Creators Update (version
  1709). You can do this by opening Bash on Windows and typing the command `lsb_release -a` and
* CMake 3.5 or later (either installed natively or built from source)
* GCC 5 or later

#### Installing Windows Subsystem for Linux (a.k.a. Bash for Windows)

1. Ensure that you have installed Windows 10 Fall Creators Update (version 1709), or later.
2. Open Start Menu
3. Type 'Developers Settings' and press Enter
4. Under 'Use developer features', ensure that 'Developer mode' is selected.
5. If prompted to reboot your computer, do not reboot.
6. Open Computer
7. Under the Computer tab, select 'Uninstall or change a program'
8. In the left-hand pane, select 'Turn Windows features on or off'
9. Scroll to the very bottom and select 'Windows Subsystem for Linux (Beta)'
10. Reboot Windows
11. Open Start Menu
12. Type 'Bash' and press Enter. Make sure that you aren't opening a mintty client such as Git Bash or Cygwin.
13. A command prompt window should open, and take you through the final part of the installation process.

#### Install developer tools

1. Type the following commands into Bash. Replace `<username>` with your Windows username.

```bash
sudo apt update && sudo apt upgrade && sudo apt update
sudo apt install build-essentials binutils gdb git flex bison texlive-full git
mkdir /mnt/c/Users/<username>/projects; cd /mnt/c/Users/<username>/projects
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make -j 4 && sudo make install
cd ..
sudo apt install ocl-icd-libopencl1 ocl-icd-dev opencl-headers clinfo lsb-core
```

2. [Download the latest OpenCL CPU driver for Ubuntu](https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime).
3. Run the following commands to install the OpenCL driver.
```bash
cd <path-to-driver>
tar -xf <driver>.tar.gz
cd <driver>
sudo ./install.sh
```
4. [Install OpenCL SDK for Windows](https://software.intel.com/en-us/intel-opencl).
5. Install [ComputeCpp-setup-v1.0]().

#### Building SYCL ParallelSTL

Once you have installed all the developer tools, follow the steps outlined for Ubuntu.

Building the documentation
----------------------------

Source code is documented using Doxygen.
To build the documentation as an HTML file, navigate to the doc
directory and run doxygen from there.

    $ cd doc
    $ doxygen

This will generate the html pages inside the doc\_output directory.

Limitations
------------

* The lambda expressions that you can pass to the algorithms have the same
restrictions as any SYCL kernel. See the SYCL specification for details
on the limitations.

* When using lambdas, the compiler needs to find a name for that expression.
To provide a lambda name, the user has to do the following:

```cpp
    cl::sycl::queue q;
    sycl::sycl_execution_policy<class SortAlgorithm3> snp(q);
    sort(snp, v.begin(), v.end(), [=](int a, int b) { return a >= b; });
```

* Be aware that some algorithms may run sequential versions if the number of
elements to be computed are not power of two. The following algorithms have
this limitation: sort, inner_product, reduce, count_if and transform_reduce.

* Refer to SYCL implementation documentation for implementation-specific
building options.

Copyright and Trademarks
------------------------

Intel and the Intel logo are trademarks of Intel Inc. AMD, the AMD Arrow
logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc.
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by
Khronos. Other names are for informational purposes only and may be trademarks
of their respective owners.
