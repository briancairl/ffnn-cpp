/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INPUT_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_INPUT_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <cstring>
#include <iostream>

// FFNN
#include <ffnn/internal/config.h>
#include <ffnn/assert.h>
#include <ffnn/layer/layer.h>

namespace ffnn
{
namespace layer
{
namespace input
{
/**
 * @brief Describes compile-time options used to set up a Input object
 */
template<size_type InputsAtCompileTime = Eigen::Dynamic>
struct options
{
  /// Total network input size
  constexpr static size_type input_size = InputsAtCompileTime;
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  ///Layer (base type) standardization
  typedef Layer<ValueType> LayerType;
};
}  // namespace input
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INPUT_COMPILE_TIME_OPTIONS_H
