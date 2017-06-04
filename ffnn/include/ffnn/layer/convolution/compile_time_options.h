/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_COMPILE_TIME_OPTIONS_H
#define FFNN_LAYER_CONVOLUTION_COMPILE_TIME_OPTIONS_H

// C++ Standard Library
#include <array>
#include <vector>
#include <type_traits>

// Boost
#include "boost/multi_array.hpp"

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/convolution/sizing.h>
#include <ffnn/layer/convolution/filter/compile_time_options.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/hidden/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
/**
 * @brief Describes compile-time options used to set up a Convolution object
 */
template<size_type HeightAtCompileTime = Eigen::Dynamic,
         size_type WidthAtCompileTime = Eigen::Dynamic,
         size_type DepthAtCompileTime = Eigen::Dynamic,
         size_type KernelHeightAtCompileTime = Eigen::Dynamic,
         size_type KernelWidthAtCompileTime = Eigen::Dynamic,
         size_type KernelCountAtCompileTime = Eigen::Dynamic,
         size_type RowStrideAtCompileTime =  1,
         size_type ColStrideAtCompileTime = -1,
         EmbeddingMode Mode = ColEmbedding>
struct options
{
  /// Data embedding mode
  constexpr static EmbeddingMode embedding_mode = Mode;

  /// Kernel kernel height
  constexpr static size_type kernel_height = KernelHeightAtCompileTime;

  /// Kernel kernel width
  constexpr static size_type kernel_width = KernelWidthAtCompileTime;

  /// Number of filter kernels
  constexpr static size_type kernel_count = KernelCountAtCompileTime;

  /// Filter stride along input rows
  constexpr static size_type row_stride = RowStrideAtCompileTime;

  /// Filter stride along input cols
  constexpr static size_type col_stride = ColStrideAtCompileTime;

  /// Input volume height
  constexpr static size_type input_height = HeightAtCompileTime;

  /// Input volume width
  constexpr static size_type input_width = WidthAtCompileTime;

  /// Input volume depth
  constexpr static size_type input_depth = DepthAtCompileTime;

  /// Output volume height
  constexpr static size_type output_height =
    output_dimension(HeightAtCompileTime, kernel_height, row_stride);

  /// Output volume width
  constexpr static size_type output_width =
    output_dimension(WidthAtCompileTime,  kernel_height, col_stride);

  /// Output volume depth
  constexpr static size_type output_depth = KernelCountAtCompileTime;

  /// Depth-embedded input height
  constexpr static size_type embedded_input_height =
    embed_dimension<Mode, ColEmbedding>(input_height, input_depth);

  /// Depth-embedded input width
  constexpr static size_type embedded_input_width =
    embed_dimension<Mode, RowEmbedding>(input_width, input_depth);

  /// Depth-embedded output height
  constexpr static size_type embedded_output_height =
    embed_dimension<Mode, ColEmbedding>(output_height, output_depth);

  /// Depth-embedded output width
  constexpr static size_type embedded_output_width =
    embed_dimension<Mode, RowEmbedding>(output_width, output_depth);
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// 2D-value mapping standardization
  typedef boost::multi_array<ValueType*, 2> ForwardMappingGridType;

  /// Filter traits type standardization
  typedef typename filter::options<
    Options::kernel_height,
    Options::kernel_width,
    Options::input_depth,
    Options::kernel_count,
    Options::embedding_mode
  > FilterOptions;

  /// Parameters type standardization
  typedef Filter<ValueType, FilterOptions> ParametersType;

  /// Compile-time Hidden layer traits
  typedef typename hidden::options<
    Options::embedded_input_height,
    Options::embedded_input_width,
    Options::embedded_output_height,
    Options::embedded_output_width,
    embed_data_order<Options::embedding_mode>(),
    embed_data_order<Options::embedding_mode>()
  > HiddenLayerOptions;

  /// Hidden layer (base type) standardization
  typedef Hidden<ValueType, HiddenLayerOptions> HiddenLayerType;
};
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_COMPILE_TIME_OPTIONS_H
