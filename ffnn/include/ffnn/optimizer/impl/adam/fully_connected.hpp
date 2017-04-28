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
class Adam<layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public GradientDescent<layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Base type standardization
  typedef typename GradientDescent<layer::Activation<ValueType, NeuronType, SizeAtCompileTime>> Base;

  /// Layer type standardization
  typedef typename layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime> LayerType;

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

  /// Bia vector type standardization
  typedef typename LayerType::BiasVector BiasVector;

  template<typename MatrixType>
  class AdamStates
  {
  public:

    void update(MatrixType& gradient, ScalarType beta1, ScalarType beta2, ScalarType eps)
    {
      // Update gradient moments
      mean_gradient_.noalias() += beta1 * (gradient - mean_gradient_);
      var_gradient_.noalias() += beta2 * (gradient - var_gradient_);

      // Compute learning rates for all weights
      gradient.noalias() = var_gradient_;
      gradient.noalias() /= (1 - beta2);
      gradient.array() += eps;
      gradient.noalias() = mean_gradient_.array() / gradient.array();
      gradient.noalias() /= (1 - beta1);
    }

    inline void initialize(SizeType rows, SizeType col)
    {
      mean_gradient_.setZero(rows, col);
      var_gradient_.setZero(rows, col);
    }

  private:
    /// Running mean of error gradient
    MatrixType mean_gradient_;

    /// Uncentered variance of error gradient
    MatrixType var_gradient_;
  };

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  Adam(ScalarType lr, ScalarType beta1 = 0.9, ScalarType beta2 = 0.999, ScalarType eps = 1e-8) :
    Base(lr),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(eps)
  {
    Base::setName("Adam[FullyConnected]");
    FFNN_ASSERT_MSG(beta1_ > 0 && beta1_ < 1, "'beta1' should be in the range (0, 1).");
    FFNN_ASSERT_MSG(beta2_ > 0 && beta2_ < 1, "'beta2' should be in the range (0, 1).");
    FFNN_ASSERT_MSG(epsilon_ > 0, "Epsilon should be > 0.");
  }
  virtual ~Adam() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  void initialize(LayerType& layer)
  {
    Base::initialize(layer);

    // Reset states
    weight_gradient_states_.initialize(layer.output_dimension_, layer.input_dimension_);
    bias_gradient_states_.initialize(layer.output_dimension_, layer.input_dimension_);
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

    // Update gradients
    weight_gradient_states_.update(Base::weight_gradient_, beta1_, beta2_, epsilon_);
    bias_gradient_states_.update(Base::bias_gradient_, beta1_, beta2_, epsilon_);

    // Update weights
    layer.w_.noalias() -= Base::lr_ * Base::weight_gradient_;
    layer.b_.noalias() -= Base::lr_ * Base::bias_gradient_;

    // Reinitialize optimizer
    Base::reset(layer);
    return true;
  }

private:
  /// Mean decay rate
  const ScalarType beta1_;

  /// Variance decay rate
  const ScalarType beta2_;

  /// Variance normalization value
  const ScalarType epsilon_;

  /// Running estimates of mean/variance of weight gradients
  AdamStates<WeightMatrix> weight_gradient_states_;

  /// Running estimates of mean/variance of bias gradients 
  AdamStates<BiasVector> bias_gradient_states_;
};
}  // namespace optimizer
}  // namespace ffnn
