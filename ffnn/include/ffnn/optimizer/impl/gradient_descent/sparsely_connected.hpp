/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_GRADIENT_DESCENT_SPARSELY_CONNECTED_HPP
#define FFNN_LAYER_IMPL_GRADIENT_DESCENT_SPARSELY_CONNECTED_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/sparsely_connected.h>

namespace ffnn
{
namespace optimizer
{
template<>
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
class GradientDescent<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public Optimizer<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Layer type standardization
  typedef typename layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::Scalar Scalar;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::OutputBlockType OutputBlockType;

  /// Bia vector type standardization
  typedef typename LayerType::BiasVectorType BiasVectorType;

  /// Input-output weight matrix
  typedef typename LayerType::WeightMatrixType WeightMatrixType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(Scalar lr) :
    Optimizer<LayerType>("GradientDescent[SparselyConnected]"),
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

    // Initialize previous input vector
    prev_input_.setZero(layer.input_shape_.size(), 1);
  }

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    // Reset weight delta
    weight_gradient_.resize(layer.output_shape_.size(), layer.input_shape_.size());

    // Reset bias delta
    bias_gradient_.setZero(layer.output_shape_.size(), 1);
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

    // Compute current weight delta
    WeightMatrixType current_weight_gradient(layer.output_shape_.size(), layer.input_shape_.size());
    for(SizeType idx = 0; idx < layer.w_.outerSize(); idx++)
    {
      for(typename WeightMatrixType::InnerIterator it(layer.w_, idx); it; ++it)
      {
        current_weight_gradient.insert(it.row(), it.col()) =
          layer.forward_error_(it.row()) * prev_input_(it.col());
      }
    }

    // Accumulate weight delta
    weight_gradient_ += current_weight_gradient;
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

    // Update weights and biases
    layer.w_ -= weight_gradient_;
    layer.b_ -= bias_gradient_;

    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  Scalar lr_;

  /// Weight matrix delta
  WeightMatrixType weight_gradient_;

  /// Total bias vector delta
  BiasVectorType bias_gradient_;

  /// Previous input
  InputBlockType prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_GRADIENT_DESCENT_SPARSELY_CONNECTED_HPP
