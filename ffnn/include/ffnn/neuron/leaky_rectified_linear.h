/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H
#define FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/neuron/neuron.h>

namespace ffnn
{
namespace neuron
{

template<typename ValueType>
struct leak_option
{
  constexpr static const ValueType leak  = 0.01;
};

/**
 * @brief A leaky-rectified linear activation unit
 */
template<typename ValueType,
         typename OptionType = leak_option<ValueType>>
class LeakyRectifiedLinear :
  public Neuron<ValueType>
{
  static_assert(OptionType::leak <  1.0, "Leak constant must be in the range [0, 1)");
  static_assert(OptionType::leak >= 0.0, "Leak constant must be in the range [0, 1)");
public:
  /**
   * @brief Setup constructor
   */
  LeakyRectifiedLinear() {}

  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void operator()(const ValueType& input, ValueType& output)
  {
    output = (input > 0) ? input : (OptionType::leak * input);
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    output = (input > 0) ? 1 : OptionType::leak;
  }
};
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_LEAKY_RECTIFIED_LINEAR_H
