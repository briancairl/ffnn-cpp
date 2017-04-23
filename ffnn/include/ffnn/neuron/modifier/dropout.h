/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_NEURON_MODIFIER_DROPOUT_H
#define FFNN_NEURON_MODIFIER_DROPOUT_H

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
         template<class> class NeuronType,
         template<class> class DistributionType,
         FFNN_OFFSET _P,
         FFNN_OFFSET _B = 100>
class Dropout :
  public Neuron<ValueType>
{
public:
  /// Default constructor
  Dropout() :
    probability_(static_cast<ValueType>(_P)/static_cast<ValueType>(_B)),
    connected_(false)
  {}

  /**
   * @brief Computes activation output
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual inline void fn(const ValueType& input, ValueType& output)
  {
    // Create distribution to draw from
    static DistributionType<ValueType> dist_;
    
    // Actiavte
    connected_ = dist_.cdf(dist_.generate()) < probability_;
    if (connected_)
    {
      Neuron<ValueType>::fn(input, output);
    }
    else
    {
      output = 0;
    }
  }

  /**
   * @brief Computes first-order activation derivative
   * @param[in] input  a scalar input value
   * @param[in,out] output  a scalar output value
   */
  virtual inline void derivative(const ValueType& input, ValueType& output) const
  {
    if (connected_)
    {
      Neuron<ValueType>::derivative(input, output);
    }
    else
    {
      output = 0;
    }
  }

private:
  /// Dropout probability
  const ValueType probability_;

  /// Flags if neuron is connected
  bool connected_;
};
}  // namespace modifier
}  // namespace neuron
}  // namespace ffnn
#endif  // FFNN_NEURON_MODIFIER_DROPOUT_H
