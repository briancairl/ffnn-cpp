/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP
#define FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/convolution.h>

namespace ffnn
{
namespace optimizer
{
template<>
template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
class GradientDescent<layer::Convolution<CONV_TARGS>>:
  public Optimizer<layer::Convolution<CONV_TARGS>>
{
public:
  /// Layer type standardization
  typedef typename layer::Convolution<CONV_TARGS> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::OutputBlockType OutputBlockType;

  /// Receptive-field type standardization
  typedef typename LayerType::ConvolutionFieldType ConvolutionFieldType;

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
  virtual void initialize(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");
    reset(layer);

    // Reset previous input
    prev_input_.setZero(layer.input_shape_.height, layer.input_shape_.width);
  }

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    // Layer sizing
    const SizeType& hs = layer.output_volume_shape_.height;
    const SizeType& ws = layer.output_volume_shape_.width;
    const SizeType& ds = layer.output_volume_shape_.depth;

    // Reset weight delta
    weight_gradient_.resize(boost::extents[hs][ws]);
    for (SizeType jdx = 0; jdx < ws; jdx++)
    {
      for (SizeType idx = 0; idx < hs; idx++)
      {
        // Create receptive field
        new (&weight_gradient_[idx][jdx]) ConvolutionVolumeType(filter_shape_, ds);
      }
    }
  }

  /**
   * @brief Computes one forward optimization update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setp was successful
   * @retval false  otherwise
   */
  virtual bool forward(LayerType& layer)
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
  virtual bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Compute and accumulate new gradient
    weight_gradient_.noalias() += layer.forward_error_ * prev_input_.transpose();
    bias_gradient_.noalias() += layer.forward_error_;

    // Back-prop error
    return true;
  }

  /**
   * @brief Applies optimization update
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  virtual bool update(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Incorporate learning rate
    weight_gradient_ *= lr_;
    bias_gradient_ *= lr_;

    // Update weights
    layer.w_.noalias() -= weight_gradient_;
    layer.b_.noalias() -= bias_gradient_;

    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  ScalarType lr_;

  /// Total weight matrix gradient
  WeightMatrixType weight_gradient_;

  /// Total bias vector delta
  BiasVectorType bias_gradient_;

  /// Previous input
  InputBlockType prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_GRADIENT_DESCENT_CONVOLUTION_HPP
