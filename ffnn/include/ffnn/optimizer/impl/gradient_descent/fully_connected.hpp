/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/fully_connected.h>

namespace ffnn
{
namespace optimizer
{
template<>
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
class GradientDescent<layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public Optimizer<layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Layer type standardization
  typedef typename layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::OutputBlockType OutputBlockType;

  /// Input-output weight matrix
  typedef typename LayerType::WeightMatrix WeightMatrix;

  /// Bia vector type standardization
  typedef typename LayerType::BiasVector BiasVector;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ScalarType lr) :
    Optimizer<LayerType>("GradientDescent[FullyConnected]"),
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
    prev_input_.setZero(layer.input_dim_.size(), 1);
  }

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    // Reset weight delta
    weight_gradient_.setZero(layer.output_dim_.size(), layer.input_dim_.size());

    // Reset bias delta
    bias_gradient_.setZero(layer.output_dim_.size(), 1);
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
    prev_input_.noalias() = *layer.input_;
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
    weight_gradient_.noalias() += (*layer.forward_error_) * prev_input_.transpose();
    bias_gradient_.noalias() += (*layer.forward_error_);

    // Compute back-propagated error
    layer.backward_error_->noalias() = layer.w_.transpose() * (*layer.forward_error_);
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
  WeightMatrix weight_gradient_;

  /// Total bias vector delta
  BiasVector bias_gradient_;

  /// Previous input
  InputBlockType prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
