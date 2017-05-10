/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
Input<ValueType, NetworkInputsAtCompileTime>::Input(SizeType network_input_size) :
  Base(ShapeType(0), ShapeType(network_input_size)),
  next_ptr_(NULL)
{}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
Input<ValueType, NetworkInputsAtCompileTime>::~Input()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Input", "Destroying [layer::Input] object <" << this->getID() << ">");
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
bool Input<ValueType, NetworkInputsAtCompileTime>::initialize()
{
  if (Base::initialize())
  {
    FFNN_DEBUG_NAMED("layer::Input",
                     "<" <<
                     Base::getID() <<
                     "> initialized as network input (net-in=" <<
                     Base::outputShape().size() <<
                     ")");
    return true;
  }
  FFNN_ERROR_NAMED("layer::Input", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
typename Input<ValueType, NetworkInputsAtCompileTime>::OffsetType
Input<ValueType, NetworkInputsAtCompileTime>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  next_ptr_ = const_cast<ValueType*>(next.getInputBuffer().data());

  // Return next offset after assigning buffer segments
  return this->outputShape().size();
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime>
template<typename NetworkInputType>
void Input<ValueType, NetworkInputsAtCompileTime>::operator<<(const NetworkInputType& input) const
{
  // Check input data size
  FFNN_ASSERT_MSG(input.size() == Base::outputShape().size(),
                  "Input data size does not match expected network input size.");

  // Copy input data to first network layer
  std::memcpy(next_ptr_, const_cast<ValueType*>(input.data()), input.size() * sizeof(ValueType));
}
}  // namespace layer
}  // namespace ffnn
