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

  typedef typename MatrixType::Scalar Scalar;

  typedef typename MatrixType::Index SizeType;

  void update(MatrixType& gradient, Scalar beta1, Scalar beta2, Scalar eps)
  {
    // Update gradient moments
    mean_ += beta1 * (gradient - mean_);
    var_  += beta2 * (gradient - var_);

    // Compute learning rates for all weights
    gradient = var_;
    gradient /= (1 - beta2);
    gradient.array() += eps;
    gradient = mean_.array() / gradient.array();
    gradient /= (1 - beta1);
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
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZATION_ADAM_STATES_HPP
