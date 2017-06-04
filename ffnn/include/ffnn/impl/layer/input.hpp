/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_INPUT_HPP
#define FFNN_LAYER_IMPL_INPUT_HPP

// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
#define TARGS ValueType, NetworkInputsAtCompileTime

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
Input<TARGS>::Input(SizeType network_input_size) :
  Base(ShapeType(0), ShapeType(network_input_size)),
  next_ptr_(NULL)
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Layer", "Network input size: " << network_input_size);
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
Input<TARGS>::~Input()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Input", "Destroying [layer::Input] object <" << this->getID() << ">");
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
bool Input<TARGS>::initialize()
{
  if (Base::initialize())
  {
    FFNN_DEBUG_NAMED("layer::Input",
                     "<" <<
                     Base::getID() <<
                     "> initialized as network input (net-in=" <<
                     Base::getOutputShape().size() <<
                     ")");
    return true;
  }
  FFNN_ERROR_NAMED("layer::Input", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
typename Input<TARGS>::OffsetType
Input<TARGS>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  next_ptr_ = const_cast<ValueType*>(next.getInputBuffer().data());

  // Return next offset after assigning buffer segments
  return this->getOutputShape().size();
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
template<typename NetworkInputType>
void Input<TARGS>::operator<<(const NetworkInputType& input) const
{
  // Check input data size
  FFNN_ASSERT_MSG(input.size() == Base::getOutputShape().size(),
                  "Input data size does not match expected network input size.");

  // Copy input data to first network layer
  std::memcpy(next_ptr_, const_cast<ValueType*>(input.data()), input.size() * sizeof(ValueType));
}
}  // namespace layer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_INPUT_HPP
