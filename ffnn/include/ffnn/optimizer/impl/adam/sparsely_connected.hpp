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
class Adam<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>:
  public GradientDescent<layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>
{
public:
  /// Base type standardization
  typedef typename GradientDescent<layer::Activation<ValueType, NeuronType, SizeAtCompileTime>> Base;

  /// Layer type standardization
  typedef typename layer::SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime> LayerType;

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

    // Reset moment matrices
    mean_gradient_.resize(layer.output_dimension_, layer.input_dimension_);
    var_gradient_.resize(layer.output_dimension_, layer.input_dimension_);
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

    // Update gradient moments
    mean_gradient_ += beta1_ * (gradient_ - mean_gradient_);
    var_gradient_  += beta2_ * (gradient_ - var_gradient_);

    // Compute learning rates for all weights
    WeightMatrix current_gradient = var_gradient_;
    current_gradient /= (1 - beta2_);
    current_gradient.array() += epsilon_;
    current_gradient = mean_gradient_.array() / current_gradient.array();
    current_gradient /= (1 - beta1_);
    current_gradient *= Base::lr_;

    // Update weights
    layer.w_ -= current_gradient;

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

  /// Running mean of error gradient
  WeightMatrix mean_gradient_;

  /// Uncentered variance of error gradient
  WeightMatrix var_gradient_;
};
}  // namespace optimizer
}  // namespace ffnn
