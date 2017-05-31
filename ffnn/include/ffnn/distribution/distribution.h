/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_DISTRIBUTION_DISTRIBUTION_H
#define FFNN_DISTRIBUTION_DISTRIBUTION_H

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
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_DISTRIBUTION_H
