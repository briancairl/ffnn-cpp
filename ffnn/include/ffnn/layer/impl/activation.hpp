/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

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
#define ACTIVATION_TARGS ValueType, NeuronType, SizeAtCompileTime
#define ACTIVATION_TARGS_ADVANCED _HiddenLayerShape
#define ACTIVATION Activation<ACTIVATION_TARGS, ACTIVATION_TARGS_ADVANCED>

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
ACTIVATION::Activation()
{}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
ACTIVATION::~Activation()
{}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
bool ACTIVATION::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Activation", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Deduce output dimensions
  Base::output_shape_ = Base::input_shape_;

  // Initialize neurons
  neurons_.resize(Base::outputShape().size());

  FFNN_DEBUG_NAMED("layer::Activation",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::inputShape().size() <<
                   ", out=" <<
                   Base::outputShape().size() <<
                   ")");

  return Base::outputShape().size() == Base::inputShape().size();
}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
bool ACTIVATION::forward()
{
  // Compute neuron outputs
  for (SizeType idx = 0; idx < Base::inputShape().size(); idx++)
  {
    neurons_[idx](Base::input_(idx), Base::output_(idx));
  }
  return true;
}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
bool ACTIVATION::backward()
{
  // Compute neuron derivatives
  Base::backward_error_.noalias() = Base::output_;
  for (SizeType idx = 0; idx < Base::outputShape().size(); idx++)
  {
    neurons_[idx].derivative(Base::input_(idx), Base::backward_error_(idx));
  }

  // Incorporate error
  Base::backward_error_.array() *= Base::forward_error_.array();
  return true;
}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
void ACTIVATION::
  save(typename ACTIVATION::OutputArchive& ar,
       typename ACTIVATION::VersionType version) const
{
  ffnn::io::signature::apply<ACTIVATION>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Activation", "Saved");
}

template<typename ValueType,
         typename NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime,
         typename _HiddenLayerShape>
void ACTIVATION::
  load(typename ACTIVATION::InputArchive& ar,
       typename ACTIVATION::VersionType version)
{
  ffnn::io::signature::check<ACTIVATION>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Activation", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
