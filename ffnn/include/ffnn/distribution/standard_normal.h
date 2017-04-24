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

template<typename ValueType>
class StandardNormal
{
public:
  /// Distribution type standardization
  typedef boost::normal_distribution<ValueType> DistributionType;

  /// Variate generator type standardization
  typedef boost::variate_generator<boost::mt19937&, DistributionType> GeneratorType;

  StandardNormal() : 
    distribution_(0.0, 1.0)
  {
    static boost::mt19937 rng;
    variate_generator_ = new GeneratorType(rng, distribution_);
  }
  virtual ~StandardNormal()
  {
    delete variate_generator_; 
  }

  /**
   * @brief Generates a random value
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  ValueType generate()
  {
    return (*variate_generator_)();
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

  GeneratorType* variate_generator_;
};
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_NORMAL_H