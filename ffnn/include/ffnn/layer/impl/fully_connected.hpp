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
Parameters::Parameters(ScalarType weight_std, ScalarType weight_mean) :
  weight_std(weight_std),
  weight_mean(weight_mean)
{
  FFNN_ASSERT_MSG(weight_std > 0, "[weight_std] should be positive");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
FullyConnected(SizeType output_dim, const Parameters& config) :
  Base(0, output_dim),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::~FullyConnected()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
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

  // Initialize weights
  if (!Base::loaded_)
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
                   Base::input_dimension_ <<
                   ", out=" <<
                   Base::output_dimension_ <<
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

  // Compute weighted outputs
  Base::output_->noalias() = w_ * (*Base::input_);
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
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

  // Set unfiormly random weight matrix
  w_.setRandom(Base::output_dimension_, Base::input_dimension_);
  w_ *= config_.weight_std;

  // Apply offset to all weights
  if (std::abs(config_.weight_mean) > 0)
  {
    w_.array() += config_.weight_mean;
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
  ar & config_.weight_std;
  ar & config_.weight_mean;

  // Save weight matrix
  ar & w_;

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
  ar & config_.weight_std;
  ar & config_.weight_mean;

  // Save weight matrix
  ar & w_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
