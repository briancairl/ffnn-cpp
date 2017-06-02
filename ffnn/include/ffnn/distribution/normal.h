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

// FFNN
#include <ffnn/assert.h>
#include <ffnn/assert.h>

namespace ffnn
{
namespace distribution
{

template<typename ValueType>
struct standard_normal
{
  constexpr static const ValueType mean  = 0.0;
  constexpr static const ValueType scale = 1.0;
};

template<typename ValueType,
         typename Options = standard_normal<ValueType>>
class Normal :
  public Distribution<ValueType>
{
public:
  /// Distribution type standardization
  typedef boost::normal_distribution<ValueType> DistributionType;

  /// Variate generator type standardization
  typedef boost::variate_generator<boost::mt19937&, DistributionType> GeneratorType;

  Normal(ValueType mean  = Options::mean,
         ValueType scale = Options::scale) : 
    distribution_(mean, scale)
  {
    FFNN_ASSERT_MSG(scale > 0, "Scale (variance) should be positive");
  }

  /**
   * @brief Generates a random value according to given distribution
   */
  ValueType generate() const
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
    const ValueType x((value - distribution_.mean()) / distribution_.sigma());
    return boost::math::erfc(-M_SQRT1_2 * x) / 2.0;
  }

private:
  DistributionType distribution_;
};

template<typename ValueType>
struct StandardNormal : Normal<ValueType> {};

}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_NORMAL_H