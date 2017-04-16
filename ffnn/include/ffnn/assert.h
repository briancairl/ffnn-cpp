/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_ASSERT_H
#define FFNN_ASSERT_H

// C++ Standard Library
#include <exception>

// FFNN
#include <ffnn/logging.h>
#ifndef FFNN_NO_ASSERT
#define FFNN_ASSERT_MSG(cond, msg)\
        if (!static_cast<bool>(cond))\
        {\
            FFNN_ERROR_NAMED("FILE:" << __FILE__ << " LN:" << __LINE__,\
                             "\n--[ASSERTION FAILED]: " << msg);\
                             throw std::runtime_error(msg);\
        }
#else
#define FFNN_ASSERT_MSG(cond, msg) {}
#endif

#define FFNN_ASSERT(cond) FFNN_ASSERT_MSG(cond, "Error")
#endif  // FFNN_ASSERT_H
