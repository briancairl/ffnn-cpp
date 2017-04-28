/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP
#define FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP

namespace ffnn
{
namespace optimizer
{
template<typename MatrixType>
class AdamStates
{
public:

  void update(MatrixType& gradient, ScalarType beta1, ScalarType beta2, ScalarType eps)
  {
    // Update gradient moments
    mean_.noalias() += beta1 * (gradient - mean_);
    var_.noalias()  += beta2 * (gradient - var_);

    // Compute learning rates for all weights
    gradient.noalias() = var_;
    gradient.noalias() /= (1 - beta2);
    gradient.array() += eps;
    gradient.noalias() = mean_.array() / gradient.array();
    gradient.noalias() /= (1 - beta1);
  }

  inline void initialize(SizeType rows, SizeType col)
  {
    mean_.setZero(rows, col);
    var_.setZero(rows, col);
  }

private:
  /// Running mean of error gradient
  MatrixType mean_;

  /// Uncentered variance of error gradient
  MatrixType var_;
};

#endif  // FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP
