/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// Boost
#include <boost/serialization/base_object.hpp>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/optimizer/none.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
Parameters::Parameters(ScalarType std_weight, ScalarType std_bias) :
  std_weight(std_weight),
  std_bias(std_bias)
{
  FFNN_ASSERT_MSG(std_weight > 0, "[std_weight] should be positive");
  FFNN_ASSERT_MSG(std_weight > 0, "[std_bias] should be positive");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
FullyConnected(SizeType output_dim, const Parameters& config) :
  Base(0, output_dim),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::~FullyConnected()
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (!Base::loaded_ && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::FullyConnected", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Set weighted-input vector
  w_input_.setZero(Base::output_dimension_, 1);

  // Initialize weights
  if (!Base::loaded_)
  {
    reset();
  }

  // Initialize neurons
  neurons_.resize(Base::output_dimension_);

  // Setup optimizer
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::FullyConnected",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::input_dimension_ <<
                   ", out=" <<
                   Base::output_dimension_ <<
                   ") [with 1 biasing input] (optimizer=" <<
                   opt_->name() <<
                   ")");
  return true;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::forward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute weighted outputs
  w_input_.noalias() = w_ * Base::input() + b_;

  // Compute neuron outputs
  for (SizeType idx = 0; idx < Base::output_dimension_; idx++)
  {
    neurons_[idx].fn(w_input_(idx), (*Base::output_)(idx));
  }
  return true;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->backward(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::reset()
{
  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Set bias vector ([-1, 1] * std(b))
  b_.setRandom(Base::output_dimension_, 1);
  b_ *= config_.std_weight;

  // Set random weight matrix ([-1, 1] * std(w))
  w_.setRandom(Base::output_dimension_, Base::input_dimension_);
  w_ *= config_.std_bias;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  save(typename FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::OutputArchive& ar,
       typename FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version) const
{
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.std_weight;
  ar & config_.std_bias;

  // Save weight matrix
  ar & w_;
  ar & b_;

  // Last weighted inputs
  ar & w_input_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Saved");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  load(typename FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::InputArchive& ar,
       typename FullyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version)
{
  Base::load(ar, version);

  // Save configuration parameters
  ar & config_.std_weight;
  ar & config_.std_bias;

  // Save weight matrix
  ar & w_;
  ar & b_;

  // Last weighted inputs
  ar & w_input_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
