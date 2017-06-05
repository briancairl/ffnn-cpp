/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_LAYER_OUTPUT_OUTPUT_HPP
#define FFNN_IMPL_LAYER_OUTPUT_OUTPUT_HPP

// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         typename Options,
         typename Extrinsics>
Output<ValueType, Options, Extrinsics>::Output(const Configuration& config) :
  BaseType(ShapeType(config.output_size_), ShapeType())
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Layer",
                            "Network output size: " << config.output_size_);
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
Output<ValueType, Options, Extrinsics>::~Output()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Output",
                            "Destroying [layer::Output] object <" << this->getID() << ">");
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
bool Output<ValueType, Options, Extrinsics>::initialize()
{
  // Abort if layer is already initialized
  if (BaseType::setupRequired() && BaseType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Output", "<" << BaseType::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  BaseType::input_shape_ = BaseType::evaluateInputSize();

  // Validate network input count
  FFNN_STATIC_RUNTIME_ASSERT_MSG (Options::output_size < 0 || BaseType::input_shape_.size() == Options::output_size,
                          "(NetworkOutputsAtCompileTime != `resolved input size`) for fixed-size layer.");

  // Do basic initialization and connect last hidden layer
  if (BaseType::initialize())
  {
    if (BaseType::connectInputLayers() != BaseType::getInputShape().size())
    {
      // Error initializing
      BaseType::initialized_ = false;
    }
    else
    {
      FFNN_DEBUG_NAMED("layer::Output",
                       "<" << BaseType::getID() <<
                       "> initialized as network output (net-out=" <<
                       BaseType::getInputShape().size() << ")");
      return true;
    }
  }
  FFNN_ERROR_NAMED("layer::Output", "<" << BaseType::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
offset_type Output<ValueType, Options, Extrinsics>::connectToForwardLayer(const BaseType& next, offset_type offset)
{
  return 0; /* do nothing */
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
template<typename NetworkOutputType>
void Output<ValueType, Options, Extrinsics>::operator>>(NetworkOutputType& output)
{
  // Check output data size
  FFNN_ASSERT_MSG(output.size() == BaseType::getInputShape().size(),
                  "Output object size does not match expected network output size.");

  // Copy output data from last network layer
  std::memcpy(const_cast<ValueType*>(output.data()),
              const_cast<ValueType*>(BaseType::input_buffer_.data()),
              BaseType::input_buffer_.size() * sizeof(ValueType));
}

template<typename ValueType,
         typename Options,
         typename Extrinsics>
template<typename NetworkTargetType>
void Output<ValueType, Options, Extrinsics>::operator<<(const NetworkTargetType& target)
{
  // Check target data size
  FFNN_ASSERT_MSG(target.size() == BaseType::getInputShape().size(),
                  "Target object size does not match expected network output size.");

  // Compute network error
  const auto n = static_cast<offset_type>(BaseType::backward_error_buffer_.size());
  for (offset_type idx = 0; idx < n; idx++)
  {
    BaseType::backward_error_buffer_.data()[idx] =
      BaseType::input_buffer_.data()[idx] - target.data()[idx];
  }
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_IMPL_LAYER_OUTPUT_OUTPUT_HPP