/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/layer/fully_connected/weights/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace fully_connected
{
/**
 * @brief Describes compile-time options used to set up a Input object
 */
template<size_type InputsAtCompileTime  = Eigen::Dynamic,
         size_type OutputsAtCompileTime = Eigen::Dynamic,
         int InputDataOrdering  = Eigen::ColMajor,
         int OutputDataOrdering = Eigen::ColMajor>
struct options
{
  /// Input count
  constexpr static size_type input_size = InputsAtCompileTime;

  /// Input data ordering
  constexpr static int input_data_ordering = InputDataOrdering;

  /// Output count
  constexpr static size_type output_size = OutputsAtCompileTime;

  /// Output data ordering
  constexpr static int output_data_ordering = OutputDataOrdering;
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Compile-time Hidden layer traits
  typedef typename hidden::options<
    Options::input_size,
    1,
    Options::output_size,
    1,
    Options::input_data_ordering,
    Options::output_data_ordering
  > WeightsOptions;

  /// Layer parameters type
  typedef Weights<ValueType, WeightsOptions> ParametersType;

  /// Compile-time Hidden layer traits
  typedef typename hidden::options<
    Options::input_size,
    1,
    Options::output_size,
    1,
    Options::input_data_ordering,
    Options::output_data_ordering
  > HiddenLayerOptions;

  /// Hidden layer (base type) standardization
  typedef Hidden<ValueType, HiddenLayerOptions> HiddenLayerType;
};
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H
