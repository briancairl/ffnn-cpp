/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_NEURON_H
#define FFNN_NEURON_NEURON_H

// FFNN
#include <ffnn/traits/shared.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A basic activation unit type
 */
template<typename ValueType>
class Neuron :
  public traits::Shared<Neuron<ValueType>>
{
public:
  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void fn(const ValueType& input, ValueType& output) const = 0;

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const = 0;
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_NEURON_H
