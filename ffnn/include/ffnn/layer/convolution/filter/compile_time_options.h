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
#include <ffnn/layer/convolution/sizing.h>
#include <ffnn/layer/convolution/filter/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
namespace filter
{
/**
 * @brief Describes compile-time options and extrinsic parameters used to set up a Filter object
 * @param HeightAtCompileTime  height of the filter kernel
 * @param WidthAtCompileTime  width of the filter kernel
 * @param DepthAtCompileTime  depth of the filter kernel
 * @param KernelCountAtCompileTime  number of kernels this filter will have
 */
template<size_type HeightAtCompileTime = Eigen::Dynamic,
         size_type WidthAtCompileTime  = Eigen::Dynamic,
         size_type DepthAtCompileTime  = Eigen::Dynamic,
         size_type KernelCountAtCompileTime = Eigen::Dynamic,
         EmbeddingMode Mode = ColEmbedding>
struct options
{
  /// Data embedding mode
  constexpr static EmbeddingMode embedding_mode = Mode;

  /// Kernel height at compile-time
  constexpr static size_type kernel_height = HeightAtCompileTime;

  /// Kernel width at compile-time
  constexpr static size_type kernel_width = WidthAtCompileTime;

  /// Kernel depth at compile-time
  constexpr static size_type kernel_depth = DepthAtCompileTime;

  /// Number of filter kernels at compile-time
  constexpr static size_type kernel_count = KernelCountAtCompileTime;

  /// Depth embedded kernel height at compile-time
  constexpr static size_type embedded_kernel_height =
    embed_dimension<embedding_mode, ColEmbedding>(kernel_height, kernel_depth);

  /// Depth embedded kernel width at compile-time
  constexpr static size_type embedded_kernel_width =
    embed_dimension<embedding_mode, RowEmbedding>(kernel_width, kernel_depth);

  /// Used to check if number of kernels in the filter is fixed
  constexpr static bool has_fixed_kernel_count = !is_dynamic(KernelCountAtCompileTime);
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Kernel type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::embedded_kernel_height,
    Options::embedded_kernel_width,
    embed_data_order<Options::embedding_mode>()
  > KernelType;

  /// Base type standardization
  typedef typename std::conditional<
    Options::has_fixed_kernel_count,
    std::array<KernelType, Options::kernel_count>,
    typename std::conditional<
      internal::traits::is_alignable_128<KernelType>::value,
      std::vector<KernelType, Eigen::aligned_allocator<KernelType>>,
      std::vector<KernelType>
    >::type
  >::type FilterBaseType;
};
}  // namespace filter
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_FILTER_COMPILE_TIME_OPTIONS_H
