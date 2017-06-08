/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <array>
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>
#include <ffnn/layer/shape.h>

namespace ffnn
{
namespace layer
{
namespace fully_connected
{
namespace weights
{
/**
 * @brief Describes compile-time options used to set up a Weights object
 */
template<size_type InputsAtCompileTime  = Eigen::Dynamic,
         size_type OutputsAtCompileTime = Eigen::Dynamic>
struct options
{
  /// FullyConnected layer input count
  constexpr static size_type input_size = InputsAtCompileTime;

  /// FullyConnected layer output count
  constexpr static size_type output_size = OutputsAtCompileTime;

  /// Used to check if dimensions are fixed
  constexpr static bool has_fixed_sizes = !is_dynamic(input_size) && !is_dynamic(output_size);
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Input block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::output_size,
    Options::input_size,
    Eigen::ColMajor
  > WeightBlockType;

  /// Output block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::output_size,
    1,
    Eigen::ColMajor
  > BiasBlockType;
};
}  // namespace weights
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_COMPILE_TIME_OPTIONS_H
