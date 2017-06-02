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

// No debug printouts in Release build
#if NDEBUG && !FFNN_ALLOW_ASSERT
#define FFNN_NO_ASSERT
#endif

#ifdef  FFNN_NO_ASSERT
/// Debug assert defintion with description
#define FFNN_ASSERT_MSG(cond, msg) FFNN_STATIC_RUNTIME_ASSERT_MSG(cond, msg)
#else
#define FFNN_ASSERT_MSG(cond, msg) (void(0))
#endif

/// Debug assert defintion
#define FFNN_ASSERT(cond) FFNN_ASSERT_MSG(cond, "Assertion Failed.")

/**
 * @brief Compile-time assertion definition
 * @note  This macro does not get evaluated-out when building in RELEASE
 */
#define FFNN_STATIC_ASSERT_MSG(cond, msg) static_assert(cond, msg)

/**
 * @brief Static runtime assertion definition
 * @note  This macro does not get evaluated-out when building in RELEASE
 */
#define FFNN_STATIC_RUNTIME_ASSERT_MSG(cond, msg)\
        {if (!static_cast<bool>(cond))\
        {\
          throw std::runtime_error(msg);\
        }}

/**
 * @brief Compile-time assert to enforce templated layer instancing pattern
 *
 *        All layer object (except for the Layer base-class) take the following template arguments:
 *        @verbatim
 *        template <typename ValueType,
 *                  ...
 *                  typename Options,
 *                  typename Extrinsics>
 *        @endverbatim
 *        where the <code>Extrinsics</code> parameter is reserved for a compile-time structure used
 *        to deduce types used by the parent layer class. Types are deduced using the <code>ValueType</code>
 *        and <code>Options</code> structure, which represent the coefficient type and compile-time
 *        sizing options, respectively.
 */
#define FFNN_ASSERT_NO_MOD_LAYER_EXTRINSICS(NS)\
        static_assert(\
          std::is_same<Extrinsics, typename NS::extrinsics<ValueType, Options>>::value,\
          "DO NOT MODIFY CLASS EXTRINSICS TEMPLATE PARAMETER!"\
        );
#endif  // FFNN_ASSERT_H
