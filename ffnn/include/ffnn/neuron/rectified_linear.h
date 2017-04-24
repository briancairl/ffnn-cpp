/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_RECTIFIED_LINEAR_H
#define FFNN_NEURON_RECTIFIED_LINEAR_H

// FFNN
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A rectified linear activation unit
 */
template<typename ValueType>
class RectifiedLinear :
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
    output = (input > 0) ? input : 0;
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    output = (input > 0) ? 1 : 0;
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_RECTIFIED_LINEAR_H
