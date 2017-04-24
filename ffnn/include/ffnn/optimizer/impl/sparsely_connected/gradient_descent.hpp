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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
class GradientDescent<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public Optimizer<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Layer type standardization
  typedef typename layer::SparselyConnected<ValueType,
                                            InputsAtCompileTime,
                                            OutputsAtCompileTime> LayerType;

  /// Scalar type standardization
  typedef typename LayerType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputVector InputVector;

  /// Matrix type standardization
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
    reset(layer);

    // Initialize previous input vector
    prev_input_.setZero(layer.input_dimension_, 1);
  }

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    // Reset weight delta
    gradient_.resize(layer.output_dimension_, layer.input_dimension_);
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

    // Compute current weight delta
    WeightMatrix current_gradient(layer.output_dimension_, layer.input_dimension_);
    for(SizeType idx = 0; idx < layer.w_.outerSize(); idx++)
    {
      for(typename WeightMatrix::InnerIterator it(layer.w_, idx); it; ++it)
      {
        current_gradient.insert(it.row(), it.col()) =
          (*layer.forward_error_)(it.row()) * prev_input_(it.col());
      }
    }

    // Accumulate weight delta
    gradient_ += current_gradient;

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

    // Update weights
    layer.w_ -= lr_ * gradient_;

    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  ScalarType lr_;

private:
  /// Weight matrix delta
  WeightMatrix gradient_;

  /// Previous input
  InputVector prev_input_;
};
}  // namespace optimizer
}  // namespace ffnn
