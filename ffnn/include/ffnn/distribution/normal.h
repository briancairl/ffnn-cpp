/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_DISTRIBUTION_NORMAL_H
#define FFNN_DISTRIBUTION_NORMAL_H

// C++ Standard Library
#include <ctime>
#include <cmath>

// Boost
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace ffnn
{
namespace distribution
{

template<typename ValueType, class _DistributionParameters>
class Normal
{
public:
  /// Distribution type standardization
  typedef boost::normal_distribution<ValueType> DistributionType;

  /// Variate generator type standardization
  typedef boost::variate_generator<boost::mt19937&, DistributionType> GeneratorType;

  Normal() : 
    distribution_(_DistributionParameters::mean,
                  _DistributionParameters::scale)
  {}

  /**
   * @brief Generates a random value
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  ValueType generate()
  {
    static boost::mt19937 rng;
    static GeneratorType gen(rng, distribution_);
    return gen();
  }

  /**
   * @brief Computes CDF of distribution at specified point
   * @param[in] value  CDF upper bound
   * @return cumulative probability
   */
  ValueType cdf(const ValueType& value) const
  {
    return 0.5 * boost::math::erfc(-M_SQRT1_2 * value);
  }

private:
  DistributionType distribution_;
};

template<typename ValueType>
struct StandardNormalParameters
{
  constexpr static const ValueType mean  = 0.0;
  constexpr static const ValueType scale = 1.0;
};

template<typename ValueType>
struct StandardNormal :
  Normal<ValueType, StandardNormalParameters<ValueType>> {};

}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_NORMAL_H