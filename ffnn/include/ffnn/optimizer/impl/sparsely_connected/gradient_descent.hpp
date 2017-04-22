/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

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
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
class GradientDescent<layer::SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public Optimizer<layer::SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Layer-type standardization
  typedef typename layer::SparselyConnected<ValueType,
                                         NeuronType,
                                         InputsAtCompileTime,
                                         OutputsAtCompileTime> LayerType;

  /// Scalar-type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size-type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix-type standardization
  typedef typename LayerType::InputVector InputVector;

  /// Matrix-type standardization
  typedef typename LayerType::OutputVector OutputVector;

  /// Input-output weight matrix
  typedef typename LayerType::WeightMatrix WeightMatrix;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ScalarType lr) :
    Optimizer<LayerType>("GradientDescent[SparselyConnected]"),
    lr_(lr)
  {}
  virtual ~GradientDescent()
  {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  virtual void initialize(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Reset weight delta
    w_delta_.resize(layer.output_dimension_, layer.input_dimension_);

    // Reset bias delta
    b_delta_.setZero(layer.output_dimension_, 1);

    // Reset previous input
    prev_input_.setZero(layer.input_dimension_, 1);
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
    prev_input_ = layer.input();
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

    using InnerIterator = typename WeightMatrix::InnerIterator;

    // Compute neuron derivatives
    OutputVector deriv = layer.output();
    for (SizeType idx = 0; idx < layer.output_dimension_; idx++)
    {
      layer.neurons_[idx].derivative(layer.w_input_(idx), deriv(idx));
    }

    // Incorporate error
    deriv.array() *= layer.forward_error_->array();

    // Compute current weight delta
    WeightMatrix w_delta_curr(layer.output_dimension_, layer.input_dimension_);
    for(SizeType idx = 0; idx < layer.w_.outerSize(); idx++)
    {
      for(InnerIterator it(layer.w_, idx); it; ++it)
      {
        w_delta_curr.insert(it.row(), it.col()) =
          deriv(it.row()) * prev_input_(it.col());
      }
    }

    // Accumulate weight delta
    w_delta_ += w_delta_curr;
    b_delta_ += deriv;

    // Compute back-propagated error
    layer.backward_error_->noalias() = layer.w_.transpose() * deriv;
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

    // Update weights
    layer.w_ -= lr_ * w_delta_;

    // Update biases
    layer.b_.noalias() -= lr_ * b_delta_;

    // Reinitialize optimizer
    initialize(layer);
    return true;
  }

protected:
  /// Learning rate
  ScalarType lr_;

private:
  /// Weight matrix delta
  WeightMatrix w_delta_;

  /// Bias vector delta
  OutputVector b_delta_;

  /// Previous input
  InputVector prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
