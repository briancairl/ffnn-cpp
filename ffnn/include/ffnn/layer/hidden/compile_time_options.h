/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_HIDDEN_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_HIDDEN_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>

namespace ffnn
{
namespace layer
{
namespace hidden
{
/**
 * @brief Describes compile-time options used to set up a Convolution object
 */
template<size_type InputHeightAtCompileTime  = Eigen::Dynamic,
         size_type InputWidthAtCompileTime   = Eigen::Dynamic,
         size_type OutputHeightAtCompileTime = Eigen::Dynamic,
         size_type OutputWidthAtCompileTime  = Eigen::Dynamic,
         int InputDataOrdering  = Eigen::ColMajor,
         int OutputDataOrdering = Eigen::ColMajor>
struct options
{
  /// Input field height
  constexpr static size_type input_height  = InputHeightAtCompileTime;

  /// Input field width
  constexpr static size_type input_width = InputWidthAtCompileTime;

  /// Input data ordering
  constexpr static int input_data_ordering = InputDataOrdering;

  /// Output field height
  constexpr static size_type output_height = OutputHeightAtCompileTime;

  /// Output field width
  constexpr static size_type output_width = OutputWidthAtCompileTime;

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
  /// Input block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::input_height,
    Options::input_width,
    Options::input_data_ordering
  > InputBlockType;

  /// Output block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::output_height,
    Options::output_width,
    Options::output_data_ordering
  > OutputBlockType;

  /// Input block-mapping type standardization
  typedef typename std::conditional<
    std::is_floating_point<ValueType>::value,
    Eigen::Map<InputBlockType, 16>,
    Eigen::Map<InputBlockType>
  >::type InputMappingType;

  /// Output block-mapping type standardization
  typedef typename std::conditional<
    std::is_floating_point<ValueType>::value,
    Eigen::Map<OutputBlockType, 16>,
    Eigen::Map<OutputBlockType>
  >::type OutputMappingType;

  ///Layer (base type) standardization
  typedef Layer<ValueType> LayerType;
};
}  // namespace hidden
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_HIDDEN_COMPILE_TIME_OPTIONS_H
