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
template<typename DistributionType>
struct is_distribution
{
  constexpr static bool value = DistributionType::IsDistribution::value;
};

template<typename ValueType>
class Distribution
{
  /// Require that ValueType is floating point
  static_assert(std::is_floating_point<ValueType>::value,
                "Distribution value representation must be a floating point type.");
public:
  typedef std::true_type IsDistribution;

  /**
   * @brief Generates a random value according to given distribution
   */
  virtual ValueType generate() const = 0;

  /**
   * @brief Computes CDF of distribution at specified point
   * @param[in] value  CDF upper bound
   * @return cumulative probability
   */
  virtual ValueType cdf(const ValueType& value) const = 0;
};
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_DISTRIBUTION_H
