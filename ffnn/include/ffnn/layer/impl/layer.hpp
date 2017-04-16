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
template<typename LayerType>
bool connect(const typename LayerType::Ptr& from, const typename LayerType::Ptr& to)
{
  // Cannot connect layers than have already been instanced in a Network
  if (from->isInitialized() || to->isInitialized())
  {
    return false;
  }
  to->prev_.push_back(from);

  if (to != from)
  {
    FFNN_DEBUG_NAMED("layer::connect", "<" << from->id() << "> connected to <" << to->id() << ">");
  }
  else
  {
    FFNN_DEBUG_NAMED("layer::connect", "<" << from->id() << "> connected to itself.");
  }
  return true;
}

template<typename ValueType>
Layer<ValueType>:: Layer(SizeType input_dim, SizeType output_dim) :
  initialized_(false),
  input_dimension_(input_dim > 0 ? input_dim : 0),
  output_dimension_(output_dim > 0 ? output_dim : 0)
{}

template<typename ValueType>
Layer<ValueType>::~Layer()
{}

template<typename ValueType>
bool Layer<ValueType>::initialize()
{
  // Abort if layer is already initialized
  if (isInitialized())
  {
    FFNN_WARN_NAMED("layer::Layer", "<" << id() << "> already initialized.");
    return false;
  }
  else if (input_dimension_)
  {
    // Allocate input buffer
    input_buffer_.resize(input_dimension_, 0);

    // Allocate backward error buffer
    backward_error_buffer_.resize(input_dimension_, 0);
  }

  // Set initialization flag
  initialized_ = true;
  return initialized_;
}

template<typename ValueType>
bool Layer<ValueType>::isInitialized()
{
  return initialized_;
}

template<typename ValueType>
typename Layer<ValueType>::SizeType Layer<ValueType>::countInputs() const
{
  SizeType count(0);
  for (const auto& layer : prev_)
  {
    count += layer->output_dimension_;
  }
  return count;
}

template<typename ValueType>
typename Layer<ValueType>::OffsetType Layer<ValueType>::connectInputLayers()
{
  // Resolve previous layer output buffers
  OffsetType offset(0);
  for (const auto& layer : prev_)
  {
    // Connect previous layers to this layer's input
    offset = layer->connectToForwardLayer(*this, offset);
  }
  return offset;
}
}  // namespace layer
}  // namespace ffnn
