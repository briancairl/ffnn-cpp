/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP
#define FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/convolution.h>

namespace ffnn
{
namespace optimizer
{
#define TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterCountAtCompileTime,\
  StrideAtCompileTime,\
  Mode,\
  _HiddenLayerShape

template<>
template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         layer::EmbeddingMode Mode,
         typename _HiddenLayerShape>
class GradientDescent<layer::Convolution<TARGS>>:
  public Optimizer<layer::Convolution<TARGS>>
{
public:
  /// Layer type standardization
  typedef typename layer::Convolution<TARGS> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Offset type standardization
  typedef typename LayerType::OffsetType OffsetType;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::OutputBlockType OutputBlockType;

  /// Parameter collection type standardization
  typedef typename LayerType::ParametersType ParametersType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ScalarType lr) :
    Optimizer<LayerType>("GradientDescent[Convolution]"),
    lr_(lr)
  {}
  virtual ~GradientDescent() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  void initialize(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Reset previous input
    prev_input_.setZero(layer.input_shape_.height, layer.input_shape_.width);

    // Reset weight delta
    new (&gradient_) ParametersType(layer.filter_shape_, layer.output_volume_shape_.depth);
    reset(layer);
  }

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  void reset(LayerType& layer)
  {
    // Set filter depth
    gradient_.filters.resize(layer.output_volume_shape_.depth);

    // Reset weight delta
    for (SizeType idx = 0; idx < layer.output_volume_shape_.height; idx++)
    {
      for (SizeType jdx = 0; jdx < layer.output_volume_shape_.width; jdx++)
      {
        gradient_.filters.setZero(layer.filter_shape_.height, layer.filter_shape_.width);
      }
    }
  }

  /**
   * @brief Computes one forward optimization update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setp was successful
   * @retval false  otherwise
   */
  bool forward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Copy current input for updating
    prev_input_.noalias() = layer.input_;
    return true;
  }

  /**
   * @brief Computes optimization step during backward propogation
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setp was successful
   * @retval false  otherwise
   */
  bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Get block dimensions
    const auto& ris = layer.parameters_.getInputShape();

    // Reset weight delta
    for (OffsetType idx = 0, hdx = 0; idx < layer.output_volume_shape_.height; idx++, hdx += layer.filter_stride_.height)
    {
      for (OffsetType jdx = 0, wdx = 0; jdx < layer.output_volume_shape_.width; jdx++, wdx += layer.filter_stride_.width)
      {
        const OffsetType n_filters(gradient_.filters.size());
        for (OffsetType kdx = 0; kdx < n_filters; kdx++)
        {
          const OffsetType iidx((Mode == layer::ColEmbedding) ? (idx * layer.output_volume_shape_.depth + kdx) : idx);
          const OffsetType jjdx((Mode == layer::RowEmbedding) ? (jdx * layer.output_volume_shape_.depth + kdx) : jdx);

          gradient_.filters[kdx].kernel += prev_input_.block(hdx, wdx, ris.height, ris.width) * layer.forward_error_(iidx, jjdx);
          gradient_.filters[kdx].bias += layer.forward_error_(iidx, jjdx);
        }
      }
    }

    // Back-prop error
    return true;
  }

  /**
   * @brief Applies optimization update
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  bool update(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Incorporate learning rate + update weights
    gradient_.filters *= lr_;
    layer.parameters_.filters -= gradient_.filters;

    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  ScalarType lr_;

  /// Total parameter gradient
  ParametersType gradient_;

  /// Previous input
  InputBlockType prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP
