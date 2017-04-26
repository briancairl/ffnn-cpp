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
Activation<ValueType, NeuronType, SizeAtCompileTime>::
Parameters::Parameters(ScalarType std_bias, ScalarType std_mean) :
  std_bias(std_bias),
  std_mean(std_mean)
{
  FFNN_ASSERT_MSG(std_bias > 0, "[std_bias] should be positive");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Activation<ValueType, NeuronType, SizeAtCompileTime>::
Activation(const Parameters& config) :
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
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
  if (!Base::loaded_ && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Activation", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Initialize biases
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

  FFNN_DEBUG_NAMED("layer::Activation",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::input_dimension_ <<
                   ", out=" <<
                   Base::output_dimension_ <<
                   ")");
  return !neurons_.empty();
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::forward()
{
  // Run optimization step
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute biased input
  b_input_.noalias() = (*Base::input_) + b_;

  // Compute neuron outputs
  for (SizeType idx = 0; idx < Base::input_dimension_; idx++)
  {
    neurons_[idx].fn(b_input_(idx), (*Base::output_)(idx));
  }
  return true;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->backward(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Activation<ValueType, NeuronType, SizeAtCompileTime>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
void Activation<ValueType, NeuronType, SizeAtCompileTime>::reset()
{
  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Set biased input
  b_input_.setZero(Base::input_dimension_, 1);

  // Set bias vector
  b_.setRandom(Base::output_dimension_, 1);
  b_ *= config_.std_bias;

  // Apply offset to all biases
  if (std::abs(config_.std_mean) > 0)
  {
    b_.array() += config_.std_mean;
  }
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
void Activation<ValueType, NeuronType, SizeAtCompileTime>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
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

  // Save configuration parameters
  ar & config_.std_bias;

  // Save weight matrix
  ar & b_;

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

  // Save configuration parameters
  ar & config_.std_bias;

  // Save weight matrix
  ar & b_;

  FFNN_DEBUG_NAMED("layer::Activation", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
