SYCL Parallel STL
==============================

This project features an implementation of the Parallel STL library
using the Khronos SYCL standard.

What is Parallel STL
-----------------------

Parallel STL is an implementation of the Technical Specification for C++ 
Extensions for Parallelism, current document number 
[N4071](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4071.htm).
This technical specification describes __a set of requirements for
implementations of an interface that computer programs written in
C++ programming language may use to invoke algorithms with parallel
execution __.
In practise, this specification aimed at the next C++ standard,
offers the opportunity to users to specify _execution policies_ to
traditional STL algorithms, which will enable the execution of
those algorithms in parallel.
The various policies can specify different kinds of parallel execution.
For example, 

```c++
std::vector<int> v = ...
// Tratidional sequential sort
std::sort(vec.begin(), vec.end());
// Explicit sequential sort
std::sort(seq, vec.begin(), vec.end());
// Explicit parallel sort
std::sort(par, vec.begin(), vec.end());
```

What is SYCL
----------------------

[SYCL](https://www.khronos.org/opencl/sycl) is a royalty-free, 
cross-platform C++ abstraction layer that builds on top of OpenCL.
SYCL enables single-source development of OpenCL applications in C++ whilst
enabling traditional host compilers to produce standard C++ code.


The SyclSTL
---------------------

SyclSTL exposes a SYCL policy on the experimental::parallel namespace
that can be passed to standard STL algorithms for them to run on SYCL.
Currently, the following STL algorithms are implemented:

* sort : Bitonic sort for ranges which size is power of two, sequential sort
otherwise.
* transform : Parallel iteration (one thread per element) on the device.

Some optimizations are implemented, for example, the ability of passing
iterators to buffers rather than STL containers to reduce the amount of
information copied in and out, and the ability of specifying a queue 
to the SYCL policy so that queue is used for the various kernels (potentially
enabling asynchronous execution of the calls).

Building the project
----------------------

The project uses CMake in order to produce build files.
Simply create a build directory and run CMake as follows:

```
$ mkdir build
$ cd build
$ cmake ../ -DSYCL_PATH=/path/to/sycl \
            -DOPENCL_ROOT_DIR=/path/to/opencl/dir
$ make
```
Usual CMake options are available (e.g. building debug or release).

If Google Mock is found in external/gmock then the unit tests are build.

