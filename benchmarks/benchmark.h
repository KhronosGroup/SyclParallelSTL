/* Copyright (c) 2015 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#include <chrono>
#include <utility>
#include <string>
#include <iostream>
#include <regex>

/**
 * output_type
 */
enum class output_type {
  STDOUT,  // Dumps output to standard output
  CSV      // Dumps output to standard output but separate fields with semicolon
};

struct benchmark_arguments {
  std::string program_name;
  output_type requestedOutput;
  bool validProgramOptions;

  void usage() {
    std::cout << " Usage: " << program_name << " [--output OUTPUT]"
              << std::endl;
    std::cout << "  --output  OUTPUT" << std::endl;
    std::cout << "        Changes the output of the benchmark, with OUTPUT: "
              << std::endl;
    std::cout << "         - CSV : Output to a CSV file " << std::endl;
    std::cout << "         - STDOUT: Output to stdout (default) " << std::endl;
  }

  benchmark_arguments(int argc, char** argv)
      : program_name(argv[0]),
        requestedOutput(output_type::STDOUT),
        validProgramOptions(true) {
    /* Match parameters */
    std::regex output_regex("--output");
    /* Check if user has specified any options */
    bool match = true;
    for (int i = 1; i < argc; i++) {
      bool matchedAnything = false;
      std::string option(argv[i]);
      if (option.size() == 0) {
        std::cerr << " Incorrect parameter " << std::endl;
        match = false;
        break;
      }
      // Check for the --output parameter
      if (std::regex_match(option, output_regex)) {
        if (i + 1 >= argc) {
          std::cerr << " Incorrect parameter " << std::endl;
          match = false;
          break;
        }
        std::string outputOption = argv[i + 1];
        std::transform(outputOption.begin(), outputOption.end(),
                       outputOption.begin(), ::tolower);
        if (outputOption == "stdout") {
          requestedOutput = output_type::STDOUT;
          matchedAnything = true;
        } else if (outputOption == "csv") {
          requestedOutput = output_type::CSV;
          matchedAnything = true;
        } else {
          match = false;
          break;
        }
        // Skip next parameter, since it was the name
        i++;
      }

      // This option is not valid
      if (!matchedAnything) {
        match = false;
        break;
      }
    }

    if (!match) {
      usage();
      validProgramOptions = false;
    }
  }
};

template <typename TimeT = std::chrono::milliseconds,
          typename ClockT = std::chrono::system_clock>
struct benchmark {
  typedef TimeT time_units_t;

  /**
   * @fn    duration
   * @brief Returns the duration (in chrono's type system) of the elapsed time
   */
  template <typename F, typename... Args>
  static TimeT duration(unsigned numReps, F func, Args&&... args) {
    TimeT dur = TimeT::zero();
    unsigned reps = 0;
    for (; reps <= numReps; reps++) {
      auto start = ClockT::now();

      func(std::forward<Args>(args)...);

      dur += std::chrono::duration_cast<TimeT>(ClockT::now() - start);
    }
    return dur / reps;
  }

  static void output_data(const std::string& short_name, int num_elems,
                          TimeT dur, output_type output = output_type::STDOUT) {
    if (output == output_type::STDOUT) {
      std::cerr << short_name << "  " << num_elems << " " << dur.count()
                << std::endl;
    } else if (output == output_type::CSV) {
      std::cerr << short_name << "," << num_elems << "," << dur.count()
                << std::endl;
    } else {
      std::cerr << " Incorrect output " << std::endl;
    }
  }
};

/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */
#define BENCHMARK_MAIN(NAME, FUNCTION, STEP_SIZE_PARAM, NUM_STEPS, REPS)      \
  int main(int argc, char* argv[]) {                                          \
    benchmark_arguments ba(argc, argv);                                       \
    if (!ba.validProgramOptions) {                                            \
      return 1;                                                               \
    }                                                                         \
    const unsigned NUM_REPS = REPS;                                           \
    const unsigned STEP_SIZE = STEP_SIZE_PARAM;                               \
    const unsigned MAX_ELEMS = STEP_SIZE * (NUM_STEPS);                       \
    for (int nelems = STEP_SIZE; nelems < MAX_ELEMS; nelems *= STEP_SIZE) {   \
      const std::string short_name = NAME;                                    \
      auto time = FUNCTION(NUM_REPS, nelems);                                 \
      benchmark<>::output_data(short_name, nelems, time, ba.requestedOutput); \
    }                                                                         \
  }
