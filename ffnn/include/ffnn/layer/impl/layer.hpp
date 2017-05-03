/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// Boost
#include <boost/serialization/base_object.hpp>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
template<typename LayerType>
bool connect(const typename LayerType::Ptr& from, const typename LayerType::Ptr& to)
{
  // Check if there is already a slot (fufill virtual connection)
  auto itr = to->prev_.find(from->getID());
  if (itr != to->prev_.end())
  {
    if (to->setupRequired() && from->setupRequired())
    {
      FFNN_DEBUG_NAMED("layer::connect", "<" << from->getID() << "> virtual connection to " <<
                                         "<" << to->getID() << "> resolved.");
      itr->second = from;
      return true;
    }
    FFNN_ERROR_NAMED("layer::connect", "Unexpected virtual connection.");
    return false;
  }

  // Cannot connect layers than have already been instanced in a Network
  if (from->isInitialized() || to->isInitialized())
  {
    return false;
  }
  else
  {
    FFNN_DEBUG_NAMED("layer::connect", "<" << from->getID() << "> connection to " <<
                                       "<" << to->getID() << "> created.");
    to->prev_.emplace(from->getID(), from);
  }
  return true;
}

template<typename ValueType>
Layer<ValueType>::Layer(const DimType& input_dim, const DimType& output_dim) :
  Layer<ValueType>::Base(input_dim, output_dim)
{}

template<typename ValueType>
Layer<ValueType>::~Layer()
{}

template<typename ValueType>
bool Layer<ValueType>::initialize()
{
  // Abort if layer is already initialized
  if (!Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Layer", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Allocate input/error buffers
  if (static_cast<SizeType>(Base::input_dim_) > 0 && input_buffer_.empty() && backward_error_buffer_.empty())
  {
    // Allocate input buffer
    input_buffer_.resize(static_cast<SizeType>(Base::input_dim_), 0);

    // Allocate backward error buffer
    backward_error_buffer_.resize(static_cast<SizeType>(Base::input_dim_), 0);
  }

  // Set initialization flag
  Base::initialized_ = true;
  return Base::initialized_;
}

template<typename ValueType>
typename Layer<ValueType>::SizeType Layer<ValueType>::evaluateInputSize() const
{
  SizeType count(0);
  for (const auto& connection : prev_)
  {
    FFNN_ASSERT_MSG(connection.second, "Virtual connection is missing layer.");
    if (!static_cast<bool>(connection.second))
    {
      FFNN_ERROR_NAMED("layer::Layer",
                       "No layer associated with virtual connection <" << connection.first << ">");
    }
    else
    {
      count += connection.second->output_dim_.size();
    }
  }
  return count;
}

template<typename ValueType>
typename Layer<ValueType>::OffsetType Layer<ValueType>::connectInputLayers()
{
  // Resolve previous layer output buffers
  OffsetType offset(0);
  for (const auto& connection : prev_)
  {
    if (!static_cast<bool>(connection.second))
    {
      FFNN_ERROR_NAMED("layer::Layer",
                       "No data for virtual connection <" << connection.first << ">");
      continue;
    }

    // Connect previous layers to this layer's input
    offset = connection.second->connectToForwardLayer(*this, offset);
  }
  return offset;
}

template<typename ValueType>
void Layer<ValueType>::save(typename Layer<ValueType>::OutputArchive& ar,
                            typename Layer<ValueType>::VersionType version) const
{
  ffnn::io::signature::apply<Layer<ValueType>>(ar);
  Base::save(ar, version);

  // Save connection information
  SizeType layer_count = prev_.size();
  ar & layer_count;
  for (const auto& connection : prev_)
  {
    ar & connection.second->getID();
  }

  FFNN_DEBUG_NAMED("layer::Layer", "Saved");
}

template<typename ValueType>
void Layer<ValueType>::load(typename Layer<ValueType>::InputArchive& ar,
                            typename Layer<ValueType>::VersionType version)
{
  ffnn::io::signature::check<Layer<ValueType>>(ar);
  Base::load(ar, version);

  // Load connection information
  SizeType layer_count;
  ar & layer_count;
  for (SizeType idx = 0; idx < layer_count; idx++)
  {
    // Load ID
    std::string id;
    ar & id;

    // Create connection with empty layer data (promise)
    prev_.emplace(id, typename Layer<ValueType>::Ptr());
  }
  FFNN_DEBUG_NAMED("layer::Layer", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
