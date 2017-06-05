/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_LAYER_INPUT_INPUT_HPP
#define FFNN_IMPL_LAYER_INPUT_INPUT_HPP

// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         typename Options,
         typename Extrinsics>
Input<ValueType, Options, Extrinsics>::Input(const Configuration& config) :
  BaseType(ShapeType(), ShapeType(config.input_size_, 1, 1)),
  config_(config),
  next_ptr_(NULL)
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Layer",
                            "Network input size: " << config.input_size_);
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
Input<ValueType, Options, Extrinsics>::~Input()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Input",
                            "Destroying [layer::Input] object <" << this->getID() << ">");
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
bool Input<ValueType, Options, Extrinsics>::initialize()
{
  if (BaseType::initialize())
  {
    FFNN_DEBUG_NAMED("layer::Input",
                     "<" <<
                     BaseType::getID() <<
                     "> initialized as network input (net-in=" <<
                     BaseType::getOutputShape().size() <<
                     ")");
    return true;
  }
  FFNN_ERROR_NAMED("layer::Input", "<" << BaseType::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
offset_type Input<ValueType, Options, Extrinsics>::connectToForwardLayer(const BaseType& next, offset_type offset)
{
  next_ptr_ = const_cast<ValueType*>(next.getInputBuffer().data());

  // Return next offset after assigning buffer segments
  return this->getOutputShape().size();
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
template<typename NetworkInputType>
void Input<ValueType, Options, Extrinsics>::operator<<(const NetworkInputType& input) const
{
  // Check input data size
  FFNN_ASSERT_MSG(input.size() == BaseType::getOutputShape().size(),
                  "Input data size does not match expected network input size.");

  // Copy input data to first network layer
  std::memcpy(next_ptr_, const_cast<ValueType*>(input.data()), input.size() * sizeof(ValueType));
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_IMPL_LAYER_INPUT_INPUT_HPP
