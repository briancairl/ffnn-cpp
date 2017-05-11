/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_MODIFIER_SOFT_DROPOUT_H
#define FFNN_NEURON_MODIFIER_SOFT_DROPOUT_H

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace neuron
{
namespace modifier
{
/**
 * @brief
 */
template<typename ValueType,
         typename NeuronType,
         typename DistributionType,
         FFNN_SIZE_TYPE _P,
         FFNN_SIZE_TYPE _B = 100>
class SoftDropout :
  public NeuronType
{
  FFNN_STATIC_ASSERT_MSG(is_neuron<NeuronType>::value, "NeuronType is not a valid Neuron object.");
public:
  /// Default constructor
  SoftDropout() :
    probability_(static_cast<ValueType>(_P)/static_cast<ValueType>(_B)),
    connection_probability_(0)
  {}

  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void operator()(const ValueType& input, ValueType& output)
  {
    // Create distribution to draw from
    static DistributionType dist;
    
    // Apply dropout/activate
    connection_probability_ = dist.cdf(dist.generate());
    output *= connection_probability_;
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual void derivative(const ValueType& input, ValueType& output) const
  {
    NeuronType::derivative(input, output);
    output *= connection_probability_;
  }

private:
  /// Dropout probability
  const ValueType probability_;

  /// Soft connection probability [0, 1]
  ValueType connection_probability_;
};
}  // namespace modifier
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_MODIFIER_SOFT_DROPOUT_H
