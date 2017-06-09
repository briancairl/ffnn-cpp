/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_OPTIMIZER_ADAM_ADAM_STATES_HPP
#define FFNN_IMPL_OPTIMIZER_ADAM_ADAM_STATES_HPP

namespace ffnn
{
namespace optimizer
{
/**
 * @brief Maintains gradient state (1st,2nd moment estimates)
 *        To be used with the Adam optimizer.
 */
template<typename LayerType>
class AdamStates
{
public:
  /// Scalar type alias
  using Scalar = typename LayerType::Scalar;

  /// Layer parameters type alias
  using ParametersType = typename LayerType::ParametersType;

  /**
   * @brief Setup constructor
   * @param beta1  1st-moment update gain parameter
   * @param beta2  2nd-moment update gain parameter
   * @param eps  Regularization parameter
   */
  AdamStates(Scalar beta1, Scalar beta2, Scalar eps) :
    inv_beta1_(1 - beta1),
    inv_beta2_(1 - beta2),
    epsilon_(eps)
  {
    FFNN_ASSERT_MSG(beta1 > 0.0, "beta1 must be in the range (0, 1)");
    FFNN_ASSERT_MSG(beta1 < 1.0, "beta1 must be in the range (0, 1)");
    FFNN_ASSERT_MSG(beta2 > 0.0, "beta2 must be in the range (0, 1)");
    FFNN_ASSERT_MSG(beta2 < 1.0, "beta2 must be in the range (0, 1)");
  }

  /**
   * @brief Updates Adam gradient states
   *
   *        Updates 1st and 2nd moment estimates for layer
   *        parameter gradients
   * @param[in,out] gradient  Layer parameter gradient
   */
  void update(ParametersType& gradient)
  {
    ParametersType t;

    // Update 1st moment
    t  = gradient;
    t -= mean_;
    t *= inv_beta1_;
    mean_ += t;
    // >> mean_' := mean_ + inv_beta1_ * (g - mean_)

    // Update 2nd moment
    gradient *= gradient;
    gradient -= var_;
    gradient *= inv_beta2_;
    var_ += gradient;
    // >> var_' := var_ + inv_beta2_ * (g * g - var_)

    // Compute learning rates for all weights
    t = mean_;
    gradient = var_;
    gradient /= inv_beta2_;
    gradient += epsilon_;
    t /= gradient;
    gradient = t;
    gradient /= inv_beta1_;
    // >> g' := mean_^ / (var_^ + eps)

  }

  /**
   * @brief Initializes Adam states from layer parameters
   * @param layer  Layer to optimize
   */
  inline void initialize(const LayerType& layer)
  {
    // Capture parameter sizings
    mean_ = layer.getParameters();
    var_  = layer.getParameters();

    // Zero out parameters
    mean_.setZero();
    var_.setZero();
  }

private:
  /// Compliment of mean decay rate
  const Scalar inv_beta1_;

  /// Compliment of variance decay rate
  const Scalar inv_beta2_;

  /// Variance normalization value
  const Scalar epsilon_;

  /// Running mean of error gradient
  ParametersType mean_;

  /// Uncentered variance of error gradient
  ParametersType var_;
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_IMPL_OPTIMIZER_ADAM_ADAM_STATES_HPP
