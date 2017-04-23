/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_LINEAR_H
#define FFNN_NEURON_LINEAR_H

// FFNN
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A linear activation unit
 * 
 *        Represents the mapping \f[ f(x) = x \f]
 */
template<typename ValueType>
class Linear :
  public Neuron<ValueType>
{
public:
  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual inline void fn(const ValueType& input, ValueType& output)
  {
    output = input;
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual inline void derivative(const ValueType& input, ValueType& output) const
  {
    output = 1;
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_LINEAR_H
