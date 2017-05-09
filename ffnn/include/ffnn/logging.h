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
/// @defgroup UnicodeColorDefinitions
/// @{
const char* HEADER = "\033[95m";
const char* BLUE = "\033[94m";
const char* GREEN = "\033[92m";
const char* WARN = "\033[93m";
const char* DEEP = "\033[96m";
const char* FAIL = "\033[91m";
const char* ENDC = "\033[0m";
const char* BOLD = "\033[1m";
const char* UNDERLINE = "\033[4m";
/// @}
}  // namespace logging

/// Named header for all <code>_NAMED</code> printouts
#define FFNN_NAME_HEADER(name) ffnn::logging::HEADER << "[" << name << "] " << ffnn::logging::ENDC

#ifdef FFNN_LOGGING_INTERNAL
/**
 * @brief Prints an named debug message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_INTERNAL_DEBUG_NAMED(name, msg)\
{std::cout << FFNN_NAME_HEADER(name) << ffnn::logging::DEEP << msg << ffnn::logging::ENDC << std::endl;}
#else
#define FFNN_INTERNAL_DEBUG_NAMED(name, msg) (void(0))
#endif

#ifndef FFNN_NO_LOGGING
/**
 * @brief Prints an named debug message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_DEBUG_NAMED(name, msg)\
{std::cout << FFNN_NAME_HEADER(name) << ffnn::logging::BLUE << msg << ffnn::logging::ENDC << std::endl;}

/**
 * @brief Prints an named warning message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_WARN_NAMED(name, msg)\
{std::cout << FFNN_NAME_HEADER(name) << ffnn::logging::WARN << msg << ffnn::logging::ENDC << std::endl;}

#else
#define FFNN_DEBUG_NAMED(name, msg) (void(0))
#define FFNN_WARN_NAMED(name, msg) (void(0))
#endif

/**
 * @brief Prints a named general info message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_INFO_NAMED(name, msg)\
{std::cout << FFNN_NAME_HEADER(name) << ffnn::logging::GREEN << msg << ffnn::logging::ENDC << std::endl;}

#if FFNN_SUPRESS_ERROR_LOGGING
#define FFNN_ERROR_NAMED(name, msg) (void(0))
#else
/**
 * @brief Prints an named error message
 * @param name  name to associate with message
 * @param msg  message to print
 */
#define FFNN_ERROR_NAMED(name, msg)\
{std::cout << FFNN_NAME_HEADER(name) << ffnn::logging::FAIL << msg << ffnn::logging::ENDC << std::endl;}
#endif

/**
 * @brief Prints an unnamed error message
 * @param name  name to associate with message
 * @param msg  message to print
 */
#define FFNN_ERROR(msg) FFNN_ERROR_NAMED("ERROR", msg)

/**
 * @brief Prints an unnamed debug message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_DEBUG(msg) FFNN_DEBUG_NAMED("DEBUG", msg)

/**
 * @brief Prints an unnamed warning message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_WARN(msg) FFNN_WARN_NAMED("-WARN", msg)

/**
 * @brief Prints an unnamed general info message
 * @param name  name to associate with message
 * @param msg  message to print
 * @warning Will not log when <code>FFNN_NO_LOGGING</code> is defined
 */
#define FFNN_INFO(msg) FFNN_INFO_NAMED("-INFO", msg)

}  // namespace ffnn
#endif  // FFNN_LOGGING_H
