/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_CONFIGURATION_H
#define FFNN_LAYER_CONVOLUTION_CONFIGURATION_H

// Boost
#include <boost/make_shared.hpp>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

#include <ffnn/distribution/distribution.h>
#include <ffnn/distribution/normal.h>

#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

#include <ffnn/layer/convolution/sizing.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
/// Layer configuration struct
template<typename LayerType,
         typename ValueType,
         typename Options,
         typename Extrinsic>
class Configuration
{
public:
  friend LayerType;

  /// Shape type standardization
  typedef typename LayerType::ShapeType ShapeType;

  /// Distribution type standardization
  typedef distribution::Distribution<ValueType> DistributionType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<LayerType> OptimizerType;

  /**
   * @brief Default constructor
   */
  Configuration() :
    input_shape_(Options::input_height, Options::input_width, Options::input_depth),
    filter_shape_(Options::kernel_height, Options::kernel_width, Options::kernel_count),
    row_stride_(Options::row_stride),
    col_stride_(Options::col_stride),
    distribution_(boost::make_shared<typename distribution::StandardNormal<ValueType>>()),
    optimizer_(boost::make_shared<typename optimizer::None<LayerType>>())
  {
    /// Try to resolve from template arguments
    resolve();
  }

  /**
   * @brief Sets layer optimization resource
   * @param optimizer  layer optimizer
   * @return *this
   */
  inline Configuration& setOptimizer(const typename OptimizerType::Ptr& optimizer)
  {
    FFNN_ASSERT_MSG(optimizer, "Optimizer resource is invalid.");

    optimizer_ = optimizer;
    return *this;
  }

  /**
   * @brief Sets layer parameter initialization distribution
   * @param distribution  value distribution resource
   * @return *this
   */
  inline Configuration& setParameterDistribution(const typename DistributionType::Ptr& distribution)
  {
    FFNN_ASSERT_MSG(distribution, "Distribution resource is invalid.");

    distribution_ = distribution;
    return *this;
  }

  /**
   * @brief Sets layer input shape
   * @param height  height of the input volume
   * @param width   width of the input volume
   * @param depth   depth of the input volume
   * @return *this
   */
  inline Configuration& setInputShape(size_type height, size_type width, size_type depth)
  {
    FFNN_ASSERT_MSG(height > 0, "Input height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Input width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Input depth must be positive.");

    input_shape_ = ShapeType(height, width, depth);
    return resolve();
  }

  /**
   * @brief Sets layer filter shape options
   * @param height  height of the filter kernel
   * @param width   width of the filter kernel
   * @param depth   number of kernels
   * @return *this
   *
   * @note   The shape specified here is <code>(H, W, N-Kernels)</code>. Each kernel actually has the
   *         dimensions <code>(H, W, D)</code>, where <code>(D)</code> is the depth of the input volume
   */
  inline Configuration& setFilterShape(size_type height, size_type width, size_type depth)
  {
    FFNN_ASSERT_MSG(height > 0, "Kernel height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Kernel width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Kernel depth must be positive.");

    filter_shape_ = ShapeType(height, width, depth);
    return resolve();
  }

  /**
   * @brief Sets filter stride options
   * @param row_stride  filter stride along rows of input volume
   * @param col_stride  filter stride along rows of input volume
   * @return *this
   *
   * @note If <code>col_stride</code> is not specified, it is defaulted to <code>row_stride</code>
   */
  inline Configuration& setStride(size_type row_stride, size_type col_stride = -1)
  {
    FFNN_ASSERT_MSG(row_stride > 0, "Sride must be positive.");

    row_stride_ = row_stride;
    col_stride_ = (col_stride <= 0) ? row_stride : col_stride;
    return resolve();
  }

private:
  /**
   * @brief Recompute sizing config from user inputs
   * @return *this
   */
  Configuration& resolve()
  {
    // Set stride shape
    stride_shape_.height = row_stride_;
    stride_shape_.width = col_stride_;
    stride_shape_.depth = input_shape_.depth;
    stride_shape_ = embed_shape_transform<Options::embedding_mode>(stride_shape_);

    // Setup output shape before depth embdedding
    output_shape_.height = output_dimension(input_shape_.height, filter_shape_.height, row_stride_);
    output_shape_.width = output_dimension(input_shape_.width,  filter_shape_.width,  col_stride_);
    output_shape_.depth = filter_shape_.depth;

    // Set depth-embedded input shape
    embedded_input_shape_ = embed_shape_transform<Options::embedding_mode>(input_shape_);

    // Set depth-embedded output shape
    embedded_output_shape_ = embed_shape_transform<Options::embedding_mode>(output_shape_);

    return *this;
  }

  /// Shape of layer input
  ShapeType input_shape_;

  /// Shape of layer output
  ShapeType output_shape_;

  /// Embdedded shape of layer input
  ShapeType embedded_input_shape_;

  /// Embdedded shape of layer output
  ShapeType embedded_output_shape_;

  /// Shape of receptive fields
  ShapeType filter_shape_;

  /// Filter stride along input rows
  size_type row_stride_;

  /// Filter stride along input cols
  size_type col_stride_;

  /// Stride between receptive fields; considers data-embedding
  ShapeType stride_shape_;

  /**
   * @brief Distribution used to initializer layer coefficients
   * @note  This will be the <code>distribution::StandardNormal</code> by default
   */
  typename DistributionType::Ptr distribution_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> by default
   */
  typename OptimizerType::Ptr optimizer_;

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
  #include <ffnn/impl/layer/convolution/configuration/serialization_class_definitions.hpp>
#endif
};
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_CONFIGURATION_H
