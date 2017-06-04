/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_NEURON_H
#define FFNN_NEURON_NEURON_H

// C++ Standard Library
#include <type_traits>

namespace ffnn
{
namespace neuron
{
template<typename NeuronType>
struct is_neuron
{
  constexpr static bool value = NeuronType::IsNeuron::value;
};

/**
 * @brief A basic activation unit type
 */
template<typename ValueType>
class Neuron
{
public:
  typedef std::true_type IsNeuron;

  typedef ValueType Scalar;

  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void operator()(const ValueType& input, ValueType& output) = 0;

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
