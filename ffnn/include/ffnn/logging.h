/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LOGGING_H
#define FFNN_LOGGING_H

// C++ Standard Library
#include <iostream>

namespace ffnn
{
namespace logging
{
  const char* HEADER = "\033[95m";
  const char* BLUE = "\033[94m";
  const char* GREEN = "\033[92m";
  const char* WARN = "\033[93m";
  const char* FAIL = "\033[91m";
  const char* ENDC = "\033[0m";
  const char* BOLD = "\033[1m";
  const char* UNDERLINE = "\033[4m";
}  // namespace logging
}  // namespace ffnn

#ifndef FFNN_NO_LOGGING
#define FFNN_HEADER(name) ffnn::logging::HEADER << "[" << name << "] " << ffnn::logging::ENDC
#define FFNN_ERROR_NAMED(name, msg)  {std::cout << FFNN_HEADER(name) << ffnn::logging::FAIL << msg << ffnn::logging::ENDC << std::endl;}
#define FFNN_DEBUG_NAMED(name, msg) {std::cout << FFNN_HEADER(name) << ffnn::logging::BLUE << msg << ffnn::logging::ENDC << std::endl;}
#define FFNN_WARN_NAMED(name, msg)  {std::cout << FFNN_HEADER(name) << ffnn::logging::WARN << msg << ffnn::logging::ENDC << std::endl;}
#else
#define FFNN_ERROR_NAMED(name, msg) {}
#define FFNN_DEBUG_NAMED(name, msg) {}
#define FFNN_WARN_NAMED(name, msg)  {}
#endif
#endif  // FFNN_LOGGING_H
