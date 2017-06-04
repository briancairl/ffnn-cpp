/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_FILTER_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_CONVOLUTION_FILTER_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <array>
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/fully_connected/sizing.h>

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
         size_type OutputsAtCompileTime = Eigen::Dynamic,
         int DataOrdering = Eigen::ColMajor>
struct options
{
  /// FullyConnected layer input count
  constexpr static size_type input_size = InputsAtCompileTime;

  /// FullyConnected layer output count
  constexpr static size_type output_size = OutputsAtCompileTime;

  /// Data ordering
  constexpr static int data_ordering = DataOrdering;

  /// Used to check if dimensions are fixed
  constexpr static bool has_fixed_sizes = !is_dynamic(InputsAtCompileTime) &&
                                          !is_dynamic(OutputsAtCompileTime);
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
    Options::data_ordering
  > WeightBlockType;

  /// Output block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::output_size,
    Options::1,
    Options::data_ordering
  > BiasBlockType;
};
}  // namespace weights
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_FILTER_COMPILE_TIME_OPTIONS_H
