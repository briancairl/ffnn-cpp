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
FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
Parameters::Parameters(ScalarType init_weight_std,
                       ScalarType init_bias_std,
                       ScalarType init_weight_mean,
                       ScalarType init_bias_mean) :
  init_weight_std(init_weight_std),
  init_bias_std(init_bias_std),
  init_weight_mean(init_weight_mean),
  init_bias_mean(init_bias_mean)
{
  FFNN_ASSERT_MSG(init_bias_std > 0, "[init_bias_std] should be positive");
  FFNN_ASSERT_MSG(init_weight_std > 0, "[init_weight_std] should be positive");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
FullyConnected(SizeType output_size, const Parameters& config) :
  Base(DimType(InputsAtCompileTime), DimType(output_size)),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::~FullyConnected()
{
  FFNN_INTERNAL_DEBUG_NAMED("FullyConnected", "Destroying [layer::FullyConnected] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::FullyConnected", "<" << Base::getID() << "> already initialized.");
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
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::FullyConnected",
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
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute weighted + biased outputs
  Base::output_.noalias() = w_ * (*Base::input_) + b_;
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");

  // Compute backward error
  Base::backward_error_->noalias() = w_.transpose() * (*Base::forward_error_);

  // Run optimizer
  return opt_->backward(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::reset()
{
  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Set uniformly random weight matrix + add biases
  w_.setRandom(Base::outputSize(), Base::inputSize());
  w_ *= config_.init_weight_std;
  if (std::abs(config_.init_weight_mean) > 0)
  {
    w_.array() += config_.init_weight_mean;
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
void FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  save(typename FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::OutputArchive& ar,
       typename FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version) const
{
  ffnn::io::signature::apply<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
void FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
  load(typename FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::InputArchive& ar,
       typename FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::VersionType version)
{
  ffnn::io::signature::check<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>>(ar);
  Base::load(ar, version);

  // Save configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
