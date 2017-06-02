/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_DISTRIBUTION_DISTRIBUTION_H
#define FFNN_DISTRIBUTION_DISTRIBUTION_H

// C++ Standard Library
#include <functional>
#include <type_traits>

// Boost
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace distribution
{
template<typename ValueType>
class Distribution
{
public:
  /// Self type standarization
  typedef Distribution<ValueType> SelfType;

  /// Self type standarization
  typedef ValueType ScalarType;

  /// Shared resource standardization
  typedef boost::shared_ptr<SelfType> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const SelfType> ConstPtr;

  /**
   * @brief Generates a random value according to given distribution
   */
  virtual ScalarType generate() const = 0;

  /**
   * @brief Computes CDF of distribution at specified point
   * @param[in] value  CDF upper bound
   * @return cumulative probability
   */
  virtual ScalarType cdf(const ScalarType& value) const = 0;
};

/**
 * @brief Sets random coefficients according to particular distribution
 * @param[in,out] x  matrix whose values to randomize
 * @param d  distribution
 */
template<typename M,
         typename DistributionType>
void setRandom(Eigen::MatrixBase<M>& x, const DistributionType& dist)
{
  // Check that scalar representation matches between objects
  using MatrixType   = typename Eigen::MatrixBase<M>;
  using M_ScalarType = typename MatrixType::Scalar;
  using D_ScalarType = typename DistributionType::ScalarType;
  static_assert(std::is_same<M_ScalarType, D_ScalarType>::value,
                "Scalar type mismatch between Eigen::MatrixBase<M> and DistributionType.");

  // Assign random values to all coefficients
  auto unaryExprSetRandomCoeff = [&dist](M_ScalarType x) -> M_ScalarType
  {
    return dist.generate();
  };
  x = x.unaryExpr(unaryExprSetRandomCoeff);
}
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_DISTRIBUTION_H
