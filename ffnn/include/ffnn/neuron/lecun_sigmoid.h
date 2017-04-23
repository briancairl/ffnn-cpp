/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_LECUN_SIGMOID_H
#define FFNN_NEURON_LECUN_SIGMOID_H

// C++ Standard Library
#include <cmath>

// FFNN
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A bipolar sigmoid activation unit scaled to prevent saturation
 * 
 *        Represents the mapping \f[ f(x) = 1.7159 * tanh(2/3 x) \f]
 *
 * @note ref: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename ValueType>
class LeCunSigmoid :
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
    output = 1.7159 * std::tanh(0.6666 * input);
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual inline void derivative(const ValueType& input, ValueType& output) const
  {
    output = 1.14382 * (1 - output * output);
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_LECUN_SIGMOID_H
