/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_FULLY_CONNECTED_HPP
#define FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_FULLY_CONNECTED_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/fully_connected.h>

namespace ffnn
{
namespace optimizer
{
template<>
template <typename ValueType,
          typename Options,
          typename Extrinsics>
class GradientDescent<layer::FullyConnected<ValueType, Options, Extrinsics>, CrossEntropy> :
  public GradientDescent_<layer::FullyConnected<ValueType, Options, Extrinsics>, CrossEntropy>
{
public:
  using LayerType = layer::FullyConnected<ValueType, Options, Extrinsics>;
  using BaseType = GradientDescent_<LayerType, CrossEntropy>;

  // Use BaseType constructors
  using BaseType::BaseType;

  bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Compute and accumulate new gradient
    this->gradient_.weights.noalias() += layer.forward_error_ * this->prev_input_.transpose();
    this->gradient_.biases.noalias() += layer.forward_error_;

    // Back-prop error
    return true;
  }
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_FULLY_CONNECTED_HPP
