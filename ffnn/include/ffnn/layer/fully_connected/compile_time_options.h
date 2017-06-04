/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/hidden/compile_time_options.h>
#include <ffnn/layer/fully_connected/weights.h>
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
         size_type OutputsAtCompileTime = Eigen::Dynamic>
struct options
{
  /// Input count
  constexpr static size_type input_size = InputsAtCompileTime;

  /// Output count
  constexpr static size_type output_size = OutputsAtCompileTime;
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Weights sizing traits
  typedef typename weights::options<
    Options::input_size,
    Options::output_size
  > WeightsOptions;

  /// Layer parameters type
  typedef Weights<ValueType, WeightsOptions> ParametersType;

  /// Compile-time Hidden layer traits
  typedef typename hidden::options<
    Options::input_size,  1,
    Options::output_size, 1,
    Eigen::ColMajor,
    Eigen::ColMajor
  > HiddenLayerOptions;

  /// Hidden layer (base type) standardization
  typedef Hidden<ValueType, HiddenLayerOptions> HiddenLayerType;
};
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_FULLY_CONNECTED_COMPILE_TIME_OPTIONS_H
