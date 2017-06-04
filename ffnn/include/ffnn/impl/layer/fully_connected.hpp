/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP
#define FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP

// Boost
#include <boost/bind.hpp>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/optimizer/none.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
template <typename ValueType,
          typename Options,
          typename Extrinsics>
FullyConnected<ValueType, Options, Extrinsics>::
FullyConnected(const Configuration& config) :
  BaseType(ShapeType(config.input_size_, 1, 1),
           ShapeType(config.output_size_, 1, 1))
{
  FFNN_INTERNAL_DEBUG_NAMED(
    "layer::FullyConnected",
    "[" << config.input_size_  << " | " << config.output_size_ << "]"
  );
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
FullyConnected<ValueType, Options, Extrinsics>::~FullyConnected()
{
  FFNN_INTERNAL_DEBUG_NAMED(
    "layer::FullyConnected",
    "Destroying [layer::FullyConnected] object <" << this->getID() << ">"
  );
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool FullyConnected<ValueType, Options, Extrinsics>::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::FullyConnected",
                    "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Initialize weights
  if (Base::setupRequired())
  {
    reset();
  }

  // Setup optimizer
  if (config_.optimizer_)
  {
    config_.optimizer_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::FullyConnected",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::getInputShape().size() <<
                   ", out=" <<
                   Base::getOutputShape().size() <<
                   ") [with 1 biasing input] (optimizer=" <<
                   config_.optimizer_->name() <<
                   ")");
  return true;
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool FullyConnected<ValueType, Options, Extrinsics>::forward()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");

  if (!config_.optimizer_->forward(*this))
  {
    return false;
  }

  // Compute weighted + biased outputs
  Base::output_.noalias() = parameters_.weights * Base::input_;
  Base::output_ += parameters_.bias;
  return true;
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool FullyConnected<ValueType, Options, Extrinsics>::backward()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");

  // Compute backward error
  Base::backward_error_.noalias() = parameters_.weights.transpose() * Base::forward_error_;

  // Run optimizer
  return config_.optimizer_->backward(*this);
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool FullyConnected<ValueType, Options, Extrinsics>::update()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");
  return config_.optimizer_->update(*this);
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void FullyConnected<ValueType, Options, Extrinsics>::reset()
{
  // Zero out connection weights and biases with appropriate size
  parameters_.setZero(Base::getInputShape().size(),
                      Base::getOutputShape().size());
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void FullyConnected<ValueType, Options, Extrinsics>::save(typename FullyConnected<ValueType, Options, Extrinsics>::OutputArchive& ar,
                                                          typename FullyConnected<ValueType, Options, Extrinsics>::VersionType version) const
{
  ffnn::internal::signature::apply<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
  Base::save(ar, version);

  // Save parameters
  ar & parameters_;

  // Save config
  ar & config_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Saved");
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void FullyConnected<ValueType, Options, Extrinsics>::load(typename FullyConnected<ValueType, Options, Extrinsics>::InputArchive& ar,
                                                          typename FullyConnected<ValueType, Options, Extrinsics>::VersionType version)
{
  ffnn::internal::signature::check<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
  Base::load(ar, version);

  // Load parameters
  ar & parameters_;

  // Load config
  ar & config_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}

}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP
