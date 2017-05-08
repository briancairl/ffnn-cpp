/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_SIGMOID_H
#define FFNN_NEURON_SIGMOID_H

// C++ Standard Library
#include <cmath>

// FFNN
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A bipolar sigmoid activation unit
 * 
 *        Represents the mapping \f[ f(x) = tanh(x) \f]
 */
template<typename ValueType>
class Sigmoid :
  public Neuron<ValueType>
{
public:
  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void operator()(const ValueType& input, ValueType& output)
  {
    output = std::tanh(input);
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    output =  (1 - output * output);
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_SIGMOID_H
