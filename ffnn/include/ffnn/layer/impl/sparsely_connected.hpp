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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
Parameters::Parameters(ScalarType connection_probability,
                       ScalarType init_weight_std,
                       ScalarType init_bias_std,
                       ScalarType init_weight_mean,
                       ScalarType init_bias_mean) :
  connection_probability(connection_probability),
  init_weight_std(init_weight_std),
  init_bias_std(init_bias_std),
  init_weight_mean(init_weight_mean),
  init_bias_mean(init_bias_mean)
{
  FFNN_ASSERT_MSG(connection_probability > 0, "[connection_probability] should be in the range (0, 1)");
  FFNN_ASSERT_MSG(connection_probability < 1, "[connection_probability] should be in the range (0, 1)");
  FFNN_ASSERT_MSG(init_bias_std > 0, "[init_bias_std] should be positive");
  FFNN_ASSERT_MSG(init_weight_std > 0, "[init_weight_std] should be positive");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
SparselyConnected(SizeType output_size, const Parameters& config) :
  Base(DimType(0), DimType(output_size)),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::~SparselyConnected()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
{
  // Deduce input dimensions
  Base::input_dim_ = DimType(Base::evaluateInputSize());

  // Abort if layer is already initialized
  if (!Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::SparselyConnected", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Initialize weights
  if (!Base::setupRequired())
  {
    reset();
  }

  // Setup optimizer
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::SparselyConnected",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::inputSize() <<
                   ", out=" <<
                   Base::outputSize() <<
                   ") [with 1 biasing input] (optimizer=" <<
                   opt_->name() <<
                   ")");
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::forward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute weighted outputs
  Base::output_->noalias() = w_ * (*Base::input_);
  Base::output_->noalias() += b_;
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->backward(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::reset()
{
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime> RandMatrix;

  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Create random value matrix
  RandMatrix random(Base::outputSize(), Base::inputSize());
  random.setRandom(Base::outputSize(), Base::inputSize());

  // Build weight matrix
  w_.resize(Base::outputSize(), Base::inputSize());
  for (SizeType idx = 0; idx < Base::outputSize(); idx++)
  {
    for (SizeType jdx = 0; jdx < Base::inputSize(); jdx++)
    {
      const ValueType p = (random(idx, jdx) + 1) / 2;
      if (p < config_.connection_probability)
      {
        w_.insert(idx, jdx) =
          config_.init_weight_mean + random(idx, jdx) * config_.init_weight_std;
      }
    }
  }

  // Set uniformly random bias matrix + add biases
  b_.setRandom(Base::outputSize(), 1);
  b_ *= config_.init_bias_std;
  if (std::abs(config_.init_bias_mean) > 0)
  {
    b_.array() += config_.init_bias_mean;
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::prune(ValueType epsilon)
{
  w_.prune(0, epsilon);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  save(typename SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::OutputArchive& ar,
       typename SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version) const
{
  ffnn::io::signature::apply<SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.connection_probability;
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  load(typename SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::InputArchive& ar,
       typename SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version)
{
  ffnn::io::signature::check<SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::load(ar, version);

  // Save configuration parameters
  ar & config_.connection_probability;
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
