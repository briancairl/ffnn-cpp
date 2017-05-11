/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_ASSERT_H
#define FFNN_ASSERT_H

// C++ Standard Library
#include <exception>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/logging.h>

// No debug printouts in Release build
#if NDEBUG && !FFNN_ALLOW_ASSERT
#define FFNN_NO_ASSERT
#endif

/// Static assertion definition
#define FFNN_STATIC_ASSERT_MSG(cond, msg) static_assert(cond, msg)

/// Static (runtime) assertion definition
#define FFNN_STATIC_RUNTIME_ASSERT_MSG(cond, msg)\
        {if (!static_cast<bool>(cond))\
        {\
            FFNN_ERROR_NAMED("FILE:" << __FILE__ << " LN:" << __LINE__, "\n>> [ASSERTION FAILED]: " << msg);\
                             throw std::runtime_error(msg);\
        }}

#ifdef  FFNN_NO_ASSERT
/// Debug assert defintion with description
#define FFNN_ASSERT_MSG(cond, msg) FFNN_STATIC_RUNTIME_ASSERT_MSG(cond, msg)
#else
#define FFNN_ASSERT_MSG(cond, msg) (void(0))
#endif

/// Debug assert defintion
#define FFNN_ASSERT(cond) FFNN_ASSERT_MSG(cond, "Assertion Failed.")
#endif  // FFNN_ASSERT_H
