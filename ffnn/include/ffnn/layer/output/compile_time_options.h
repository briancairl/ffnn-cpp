/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OUTPUT_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_OUTPUT_COMPILE_TIME_OPTIONS_H

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
namespace output
{
/**
 * @brief Describes compile-time options used to set up a Output object
 */
template<size_type OutputsAtCompileTime = Eigen::Dynamic>
struct options
{
  /// Total network output size
  constexpr static size_type output_size = OutputsAtCompileTime;
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
}  // namespace output
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_OUTPUT_COMPILE_TIME_OPTIONS_H
