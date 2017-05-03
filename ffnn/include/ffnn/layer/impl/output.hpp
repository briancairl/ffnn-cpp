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
template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<ValueType, NetworkOutputsAtCompileTime>::Output() :
  Base(DimType(NetworkOutputsAtCompileTime), DimType(0))
{}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<ValueType, NetworkOutputsAtCompileTime>::~Output()
{}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
bool Output<ValueType, NetworkOutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (!Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Output", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  Base::input_dim_ = Base::evaluateInputSize();

  // Validate network input count
  FFNN_STATIC_ASSERT_MSG (NetworkOutputsAtCompileTime < 0 || Base::input_dim_.size() == NetworkOutputsAtCompileTime,
                          "(NetworkOutputsAtCompileTime != `resolved input size`) for fixed-size layer.");

  // Do basic initialization and connect last hidden layer
  if (Base::initialize())
  {
    if (Base::connectInputLayers() != Base::inputSize())
    {
      // Error initializing
      Base::initialized_ = false;
    }
    else
    {
      FFNN_DEBUG_NAMED("layer::Output",
                       "<" << Base::getID() <<
                       "> initialized as network output (net-out=" <<
                       Base::inputSize() << ")");
      return true;
    }
  }
  FFNN_ERROR_NAMED("layer::Output", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
typename Output<ValueType, NetworkOutputsAtCompileTime>::OffsetType
Output<ValueType, NetworkOutputsAtCompileTime>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  return 0; /* do nothing */
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkOutputType>
void Output<ValueType, NetworkOutputsAtCompileTime>::operator>>(NetworkOutputType& output)
{
  // Check output data size
  FFNN_ASSERT_MSG(output.size() == Base::inputSize(),
                  "Output object size does not match expected network output size.");

  // Copy output data from last network layer
  std::memcpy(const_cast<ValueType*>(output.data()),
              const_cast<ValueType*>(Base::input_buffer_.data()),
              Base::input_buffer_.size() * sizeof(ValueType));
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkTargetType>
void Output<ValueType, NetworkOutputsAtCompileTime>::operator<<(const NetworkTargetType& target)
{
  // Check target data size
  FFNN_ASSERT_MSG(target.size() == Base::inputSize(),
                  "Target object size does not match expected network output size.");

  // Compute network error
  for (SizeType idx = 0; idx < Base::backward_error_buffer_.size(); idx++)
  {
    Base::backward_error_buffer_.data()[idx] = Base::input_buffer_.data()[idx] - target.data()[idx];
  }
}
}  // namespace layer
}  // namespace ffnn
