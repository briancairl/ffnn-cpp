/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/activation.h>

namespace ffnn
{
namespace optimizer
{
template<>
template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
class GradientDescent<layer::Activation<ValueType, NeuronType, SizeAtCompileTime>>:
  public Optimizer<layer::Activation<ValueType, NeuronType, SizeAtCompileTime>>
{
public:
  /// Layer type standardization
  typedef typename layer::Activation<ValueType, NeuronType, SizeAtCompileTime> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputVector InputVector;

  /// Matrix type standardization
  typedef typename LayerType::BiasVector BiasVector;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ScalarType lr) :
    Optimizer<LayerType>("GradientDescent[Activation]"),
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
  }

  /**
   * @brief Resets persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    // Reset bias delta
    gradient_.setZero(layer.output_dimension_, 1);
  }

  /**
   * @brief Computes one forward optimization update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  virtual bool forward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");
    return true;
  }

  /**
   * @brief Computes optimization step during backward propagation
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  virtual bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Compute neuron derivatives
    layer.backward_error_->noalias() = (*layer.output_);
    for (SizeType idx = 0; idx < layer.output_dimension_; idx++)
    {
      layer.neurons_[idx].derivative(layer.b_input_(idx), (*layer.backward_error_)(idx));
    }

    // Incorporate error
    layer.backward_error_->array() *= layer.forward_error_->array();

    // Accumulate weight delta
    gradient_.noalias() += (*layer.backward_error_);
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
    gradient_ *= lr_;

    // Update biases
    layer.b_.noalias() -= gradient_;
  
    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  ScalarType lr_;

  /// Bias vector delta
  BiasVector gradient_;
};
}  // namespace optimizer
}  // namespace ffnn
