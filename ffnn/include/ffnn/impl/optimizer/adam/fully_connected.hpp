/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/fully_connected.h>
#include <ffnn/optimizer/impl/adam/adam_states.hpp>

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
  /// Layer type standardization
  typedef typename layer::FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime> LayerType;

  /// Base type standardization
  typedef GradientDescent<LayerType> Base;

  /// Scalar type standardization
  typedef typename LayerType::Scalar Scalar;

  /// Size type standardization
  typedef typename LayerType::SizeType SizeType;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::OutputBlockType OutputBlockType;

  /// Input-output weight matrix
  typedef typename LayerType::WeightMatrixType WeightMatrixType;

  /// Bia vector type standardization
  typedef typename LayerType::BiasVectorType BiasVectorType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  Adam(Scalar lr, Scalar beta1 = 0.9, Scalar beta2 = 0.999, Scalar eps = 1e-8) :
    Base(lr),
    gradient_states_(beta1, beta2, eps)
  {
    Base::setName("Adam");
    FFNN_ASSERT_MSG(beta1 > 0 && beta1 < 1, "'beta1' should be in the range (0, 1).");
    FFNN_ASSERT_MSG(beta2 > 0 && beta2 < 1, "'beta2' should be in the range (0, 1).");
    FFNN_ASSERT_MSG(eps > 0, "Epsilon should be > 0.");
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
    weight_gradient_states_.initialize(layer.output_shape_.size(), layer.input_shape_.size());
    bias_gradient_states_.initialize(layer.output_shape_.size(), layer.input_shape_.size());
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

    // Reinitialize optimizer
    Base::update(layer);
    return true;
  }

private:
  /// Mean decay rate
  const Scalar beta1_;

  /// Variance decay rate
  const Scalar beta2_;

  /// Variance normalization value
  const Scalar epsilon_;

  /// Running estimates of mean/variance of weight gradients
  AdamStates<WeightMatrixType> gradient_states_;
};
}  // namespace optimizer
}  // namespace ffnn
