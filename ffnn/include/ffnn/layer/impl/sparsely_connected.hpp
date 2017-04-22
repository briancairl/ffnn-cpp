/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
Parameters::Parameters(ScalarType std_weight, ScalarType std_bias, ScalarType connection_probability) :
  std_weight(std_weight),
  std_bias(std_bias),
  connection_probability(connection_probability)
{
  FFNN_ASSERT_MSG(std_weight > 0, "[std_weight] should be positive");
  FFNN_ASSERT_MSG(std_bias > 0, "[std_bias] should be positive");
  FFNN_ASSERT_MSG(connection_probability > 0, "[connection_probability] should be in range (0, 1)");
  FFNN_ASSERT_MSG(connection_probability < 1, "[connection_probability] should be in range (0, 1)");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
SparselyConnected(SizeType output_dim, const Parameters& config) :
  Base(0, output_dim),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::~SparselyConnected()
{}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (!Base::loaded_ && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::SparselyConnected", "<" << Base::getID() << "> already initialized.");
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

  FFNN_DEBUG_NAMED("layer::SparselyConnected",
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
bool SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::forward()
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
bool SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->backward(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::reset()
{
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime> RandMatrix;

  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Set bias vector ([-1, 1] * std(b))
  b_.setRandom(Base::output_dimension_, 1);
  b_ *= config_.std_bias;

  // Create random value matrix
  RandMatrix random(Base::output_dimension_, Base::input_dimension_);
  random.setRandom(Base::output_dimension_, Base::input_dimension_);

  // Build weight matrix
  w_.resize(Base::output_dimension_, Base::input_dimension_);
  for (SizeType idx = 0; idx < Base::output_dimension_; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::input_dimension_; jdx++)
    {
      const ValueType p = (random(idx, jdx) + 1) / 2;
      if (p < config_.connection_probability)
      {
        w_.insert(idx, jdx) = random(idx, jdx) * config_.std_weight;
      }
    }
  }
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::prune(ValueType epsilon)
{
  w_.prune(0, epsilon);
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  save(typename SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::OutputArchive& ar,
       typename SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version) const
{
  ffnn::io::signature::apply<SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.connection_probability;
  ar & config_.std_weight;
  ar & config_.std_bias;

  // Save weight matrix
  ar & w_;
  ar & b_;

  // Last weighted inputs
  ar & w_input_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Saved");
}

template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::
  load(typename SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::InputArchive& ar,
       typename SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version)
{
  ffnn::io::signature::check<SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::load(ar, version);

  // Save configuration parameters
  ar & config_.connection_probability;
  ar & config_.std_weight;
  ar & config_.std_bias;

  // Save weight matrix
  ar & w_;
  ar & b_;

  // Last weighted inputs
  ar & w_input_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
