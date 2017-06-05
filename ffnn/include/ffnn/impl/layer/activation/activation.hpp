/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_LAYER_ACTIVATION_ACTIVATION_HPP
#define FFNN_IMPL_LAYER_ACTIVATION_ACTIVATION_HPP

// C++ Standard library
#include <ctime>
#include <cstring>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/optimizer/none.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
Activation<ValueType, NeuronType, Options, Extrinsics>::
Activation(const Configuration& config) :
  BaseType(ShapeType(config.input_size_, 1, 1), ShapeType(config.output_size_, 1, 1))
{}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
Activation<ValueType, NeuronType, Options, Extrinsics>::~Activation()
{}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
bool Activation<ValueType, NeuronType, Options, Extrinsics>::initialize()
{
  // Abort if layer is already initialized
  if (BaseType::setupRequired() && BaseType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Activation", "<" << BaseType::getID() << "> already initialized.");
    return false;
  }
  else if (!BaseType::initialize())
  {
    return false;
  }

  // Deduce output dimensions
  BaseType::output_shape_ = BaseType::input_shape_;

  // Initialize neurons
  reset();

  FFNN_DEBUG_NAMED("layer::Activation",
                   "<" <<
                   BaseType::getID() <<
                   "> initialized as (in=" <<
                   BaseType::getInputShape().size() <<
                   ", out=" <<
                   BaseType::getOutputShape().size() <<
                   ")");

  return BaseType::getOutputShape().size() ==
         BaseType::getInputShape().size();
}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
bool Activation<ValueType, NeuronType, Options, Extrinsics>::forward()
{
  // Compute neuron outputs
  for (offset_type idx = 0; idx < BaseType::getInputShape().size(); idx++)
  {
    neurons_[idx](BaseType::input_(idx), BaseType::output_(idx));
  }
  return true;
}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
bool Activation<ValueType, NeuronType, Options, Extrinsics>::backward()
{
  // Compute neuron derivatives
  BaseType::backward_error_.noalias() = BaseType::output_;
  for (offset_type idx = 0; idx < BaseType::getOutputShape().size(); idx++)
  {
    neurons_[idx].derivative(BaseType::input_(idx), BaseType::backward_error_(idx));
  }

  // Incorporate error
  BaseType::backward_error_.array() *= BaseType::forward_error_.array();
  return true;
}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
template<bool T>
typename std::enable_if<T>::type
Activation<ValueType, NeuronType, Options, Extrinsics>::reset()
{}
template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
template<bool T>
typename std::enable_if<!T>::type
Activation<ValueType, NeuronType, Options, Extrinsics>::reset()
{
  neurons_.resize(BaseType::getOutputShape().size());
}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
void Activation<ValueType, NeuronType, Options, Extrinsics>::
  save(OutputArchive& ar, VersionType version) const
{
  ffnn::internal::signature::apply<SelfType>(ar);
  BaseType::save(ar, version);

  FFNN_DEBUG_NAMED("layer::Activation", "Saved");
}

template<typename ValueType,
         typename NeuronType,
         typename Options,
         typename Extrinsics>
void Activation<ValueType, NeuronType, Options, Extrinsics>::
  load(InputArchive& ar, VersionType version)
{
  ffnn::internal::signature::check<SelfType>(ar);
  BaseType::load(ar, version);

  FFNN_DEBUG_NAMED("layer::Activation", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_IMPL_LAYER_ACTIVATION_ACTIVATION_HPP
