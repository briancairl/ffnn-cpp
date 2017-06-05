/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_HPP
#define FFNN_IMPL_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_HPP

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
  BaseType(ShapeType(config.input_size_, 1, 1), ShapeType(config.output_size_, 1, 1))
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
  if (BaseType::setupRequired() && BaseType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::FullyConnected",
                    "<" << BaseType::getID() << "> already initialized.");
    return false;
  }
  else if (!BaseType::initialize())
  {
    return false;
  }

  // Initialize weights
  if (BaseType::setupRequired())
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
                   BaseType::getID() <<
                   "> initialized as (in=" <<
                   BaseType::getInputShape().size() <<
                   ", out=" <<
                   BaseType::getOutputShape().size() <<
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
  BaseType::output_.noalias() = parameters_.weights * BaseType::input_;
  BaseType::output_ += parameters_.biases;
  return true;
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool FullyConnected<ValueType, Options, Extrinsics>::backward()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");

  // Compute backward error
  BaseType::backward_error_.noalias() = parameters_.weights.transpose() * BaseType::forward_error_;

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
  parameters_.setZero(BaseType::getInputShape().size(),
                      BaseType::getOutputShape().size());
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void FullyConnected<ValueType, Options, Extrinsics>::save(OutputArchive& ar, VersionType version) const
{
  ffnn::internal::signature::apply<SelfType>(ar);
  BaseType::save(ar, version);

  // Save parameters
  ar & parameters_;

  // Save config
  ar & config_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Saved");
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void FullyConnected<ValueType, Options, Extrinsics>::load(InputArchive& ar, VersionType version)
{
  ffnn::internal::signature::check<SelfType>(ar);
  BaseType::load(ar, version);

  // Load parameters
  ar & parameters_;

  // Load config
  ar & config_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}

}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_IMPL_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_HPP
