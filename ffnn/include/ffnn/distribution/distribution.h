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
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>

namespace ffnn
{
namespace distribution
{
template<typename ValueType>
class Distribution
{
public:
  /// Scalar type standarization
  typedef ValueType Scalar;

  /// Self type standarization
  typedef Distribution<ValueType> SelfType;

  /// Shared resource standardization
  typedef boost::shared_ptr<SelfType> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const SelfType> ConstPtr;

  /**
   * @brief Generates a random value according to given distribution
   */
  virtual Scalar generate() const = 0;

  /**
   * @brief Computes CDF of distribution at specified point
   * @param[in] value  CDF upper bound
   * @return cumulative probability
   */
  virtual Scalar cdf(const Scalar& value) const = 0;
};

/**
 * @brief Sets random coefficients according to particular distribution
 * @param[in,out] x  matrix whose values to randomize
 * @param d  distribution
 */
template<typename BlockType,
         typename DistributionType>
void setRandom(Eigen::MatrixBase<BlockType>& x, const DistributionType& dist)
{
  static_assert(internal::traits::is_distribution<DistributionType>::value,
                "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

  static_assert(std::is_same<typename DistributionType::Scalar,
                             typename Eigen::MatrixBase<BlockType>::Scalar>::value,
                "SCALAR TYPE MISMATCH BETWEEN [Eigen::MatrixBase<BlockType>::Scalar] AND [DistributionType::Scalar]!");

  // Assign random values to all coefficients
  auto unaryExprSetRandomCoeff = [&dist](typename DistributionType::Scalar x)
    -> typename DistributionType::Scalar
  {
    return dist.generate();
  };
  x = x.unaryExpr(unaryExprSetRandomCoeff);
}
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_DISTRIBUTION_H
