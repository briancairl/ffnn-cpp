/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_SOFT_SIGN_H
#define FFNN_NEURON_SOFT_SIGN_H

// C++ Standard Library
#include <cmath>

// FFNN
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A soft-sign activation unit
 * 
 *        Represents the mapping \f[ f(x) = 1 / (1 + |x|^{2}) \f]
 */
template<typename ValueType>
class SoftSign :
  public Neuron<ValueType>
{
public:
  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void fn(const ValueType& input, ValueType& output)
  {
    output = input / (1 + std::abs(input));
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    const ValueType v = (1 + std::abs(input));
    output = 1 / (v * v);
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_SOFT_SIGN_H
