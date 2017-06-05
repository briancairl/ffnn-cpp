/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_ACTIVATION_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_ACTIVATION_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <array>
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>
#include <ffnn/layer/hidden/hidden.h>
#include <ffnn/layer/hidden/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace activation
{
/**
 * @brief Describes compile-time options used to set up a Convolution object
 */
template<size_type SizeAtCompileTime = Eigen::Dynamic>
struct options
{
  /// Input field height
  constexpr static size_type input_size = SizeAtCompileTime;

  /// Output field height
  constexpr static size_type output_size = SizeAtCompileTime;

  /// Used to check if dimensions are fixed
  constexpr static bool has_fixed_sizes = !is_dynamic(SizeAtCompileTime);
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename NeuronType,
         typename Options>
struct extrinsics
{
  static_assert(internal::traits::is_neuron<NeuronType>::value,
                "[NeuronType] DOES NOT FUFILL THE NEURON CONCEPT!");

  /// Nueron block type standardization
  typedef typename std::conditional<
    Options::has_fixed_sizes,
    std::array<NeuronType, Options::input_size>,
    std::vector<NeuronType>
  >::type NeuronBlockType;

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
}  // namespace activation
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_ACTIVATION_COMPILE_TIME_OPTIONS_H
