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
template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Activation<ValueType, NeuronType, SizeAtCompileTime>::Activation()
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Activation<ValueType, NeuronType, SizeAtCompileTime>::~Activation()
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::initialize()
{
  // This layer has equal inputs and outputs
  Base::output_dimension_ = Base::countInputs();

  // Abort if layer is already initialized
  if (Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Activation", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Initialize neurons
  neurons_.resize(Base::output_dimension_);

  FFNN_DEBUG_NAMED("layer::Activation",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::input_dimension_ <<
                   ", out=" <<
                   Base::output_dimension_ <<
                   ")");

  return Base::output_dimension_ == Base::input_dimension_;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::forward()
{
  // Compute neuron outputs
  for (SizeType idx = 0; idx < Base::input_dimension_; idx++)
  {
    neurons_[idx].fn((*Base::input_)(idx), (*Base::output_)(idx));
  }
  return true;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::backward()
{
  // Compute neuron derivatives
  Base::backward_error_->noalias() = *Base::output_;
  for (SizeType idx = 0; idx < Base::output_dimension_; idx++)
  {
    neurons_[idx].derivative((*Base::input_)(idx), (*Base::backward_error_)(idx));
  }

  // Incorporate error
  Base::backward_error_->array() *= Base::forward_error_->array();
  return true;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
void Activation<ValueType, NeuronType, SizeAtCompileTime>::
  save(typename Activation<ValueType, NeuronType, SizeAtCompileTime>::OutputArchive& ar,
       typename Activation<ValueType, NeuronType, SizeAtCompileTime>::VersionType version) const
{
  ffnn::io::signature::apply<Activation<ValueType, NeuronType, SizeAtCompileTime>>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Activation", "Saved");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
void Activation<ValueType, NeuronType, SizeAtCompileTime>::
  load(typename Activation<ValueType, NeuronType, SizeAtCompileTime>::InputArchive& ar,
       typename Activation<ValueType, NeuronType, SizeAtCompileTime>::VersionType version)
{
  ffnn::io::signature::check<Activation<ValueType, NeuronType, SizeAtCompileTime>>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Activation", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
