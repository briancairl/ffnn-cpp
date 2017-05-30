/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_OUTPUT_HPP
#define FFNN_LAYER_IMPL_OUTPUT_HPP

// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
#define TARGS ValueType, NetworkOutputsAtCompileTime

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<TARGS>::Output() :
  Base(ShapeType(NetworkOutputsAtCompileTime), ShapeType(0))
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Layer", "Network output size (compile-time): " << NetworkOutputsAtCompileTime);
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<TARGS>::~Output()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Output", "Destroying [layer::Output] object <" << this->getID() << ">");
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
bool Output<TARGS>::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Output", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  Base::input_shape_ = Base::evaluateInputSize();

  // Validate network input count
  FFNN_STATIC_RUNTIME_ASSERT_MSG (NetworkOutputsAtCompileTime < 0 || Base::input_shape_.size() == NetworkOutputsAtCompileTime,
                          "(NetworkOutputsAtCompileTime != `resolved input size`) for fixed-size layer.");

  // Do basic initialization and connect last hidden layer
  if (Base::initialize())
  {
    if (Base::connectInputLayers() != Base::getInputShape().size())
    {
      // Error initializing
      Base::initialized_ = false;
    }
    else
    {
      FFNN_DEBUG_NAMED("layer::Output",
                       "<" << Base::getID() <<
                       "> initialized as network output (net-out=" <<
                       Base::getInputShape().size() << ")");
      return true;
    }
  }
  FFNN_ERROR_NAMED("layer::Output", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
typename Output<TARGS>::OffsetType
Output<TARGS>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  return 0; /* do nothing */
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkOutputType>
void Output<TARGS>::operator>>(NetworkOutputType& output)
{
  // Check output data size
  FFNN_ASSERT_MSG(output.size() == Base::getInputShape().size(),
                  "Output object size does not match expected network output size.");

  // Copy output data from last network layer
  std::memcpy(const_cast<ValueType*>(output.data()),
              const_cast<ValueType*>(Base::input_buffer_.data()),
              Base::input_buffer_.size() * sizeof(ValueType));
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkTargetType>
void Output<TARGS>::operator<<(const NetworkTargetType& target)
{
  // Check target data size
  FFNN_ASSERT_MSG(target.size() == Base::getInputShape().size(),
                  "Target object size does not match expected network output size.");

  // Compute network error
  for (SizeType idx = 0; idx < Base::backward_error_buffer_.size(); idx++)
  {
    Base::backward_error_buffer_.data()[idx] = Base::input_buffer_.data()[idx] - target.data()[idx];
  }
}
}  // namespace layer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_OUTPUT_HPP