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
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace ffnn
{
namespace distribution
{

template<typename ValueType>
class Normal
{
public:
  /// Distribution type standardization
  typedef boost::normal_distribution<ValueType> DistributionType;

  /// Variate generator type standardization
  typedef boost::variate_generator<boost::mt19937&, DistributionType> GeneratorType;

  Normal(ValueType mean, ValueType scale) : 
    distribution_(mean, scale)
  {
    static boost::mt19937 rng;
    if (!variate_generator_)
    {
      variate_generator_ = boost::make_shared<GeneratorType>(rng, distribution_);
    }
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
  static boost::shared_ptr<GeneratorType> variate_generator_;
};


template<typename ValueType>
struct StandardNormal : Normal<ValueType>
{
  /// Base-type aliad
  using Base = Normal<ValueType>;

  /// Distribution type standardization
  typedef typename Base::DistributionType DistributionType;

  /// Variate generator type standardization
  typedef typename Base::GeneratorType GeneratorType;

  StandardNormal() : 
    Base(0.0, 1.0)
  {}
};
}  // namespace distribution
}  // namespace ffnn
#endif  // FFNN_DISTRIBUTION_NORMAL_H