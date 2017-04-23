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
  /**
   * @brief Generates a random value
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual ValueType generate() = 0;

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