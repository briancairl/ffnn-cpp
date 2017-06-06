/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP
#define FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP

namespace ffnn
{
namespace optimizer
{
template<typename ValueType, typename LayerType>
class AdamStates
{
public:
  AdamStates(ValueType beta1, ValueType beta2, ValueType eps) :
    inv_beta1_(1 - beta1),
    inv_beta2_(1 - beta2),
    epsilon_(eps)
  {}

  void update(ParametersType& gradient)
  {
    // Update gradient moments
    mean_ += beta1 * (gradient - mean_);
    var_  += beta2 * (gradient - var_);

    // Compute learning rates for all weights
    ParametersType tmp;
    gradient = var_;
    gradient /= inv_beta2_;
    gradient += epsilon_;
    tmp = mean_;
    tmp /= gradient;
    gradient = tmp;
    gradient /= inv_beta1_;
  }

  inline void initialize(const LayerType& layer)
  {
    mean_ = layer.getParameters();
    mean_.setZero();

    var_  = layer.getParameters();
    var_.setZero();
  }

private:
  /// Mean decay rate
  const ValueType inv_beta1_;

  /// Variance decay rate
  const ValueType inv_beta2_;

  /// Variance normalization value
  const ValueType epsilon_;

  /// Running mean of error gradient
  ParametersType mean_;

  /// Uncentered variance of error gradient
  ParametersType var_;
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP
