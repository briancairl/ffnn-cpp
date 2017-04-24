/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H
#define FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/config/global.h>
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{
/**
 * @brief A leaky-rectified linear activation unit
 */
template<typename ValueType,
         FFNN_SIZE_TYPE _P,
         FFNN_SIZE_TYPE _B = 100>
class LeakyRectifiedLinear :
  public Neuron<ValueType>
{
public:
  /**
   * @brief Setup constructor
   * @param leak_factor input to leak with (input < 0) 
   */
  LeakyRectifiedLinear() :
    leak_factor_(static_cast<ValueType>(_P)/static_cast<ValueType>(_B))
  {
    FFNN_ASSERT_MSG((leak_factor > 0 && leak_factor <= 1),
                    "Leak constant is not in the range [0, 1]");
  }

  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void fn(const ValueType& input, ValueType& output)
  {
    output = (input > 0) ? input : (leak_factor_ * input);
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    output = (input > 0) ? 1 : leak_factor_;
  }

protected:
  /// Factor in the range [0, 1] to leak when (input < 0)
  const ValueType leak_factor_;
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H
