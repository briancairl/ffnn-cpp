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
Parameters::Parameters(ScalarType weight_std,
                       ScalarType weight_mean,
                       ScalarType connection_probability) :
  weight_std(weight_std),
  weight_mean(weight_mean),
  connection_probability(connection_probability)
{
  FFNN_ASSERT_MSG(weight_std > 0, "[weight_std] should be positive");
  FFNN_ASSERT_MSG(connection_probability > 0, "[connection_probability] should be in range (0, 1)");
  FFNN_ASSERT_MSG(connection_probability < 1, "[connection_probability] should be in range (0, 1)");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::
SparselyConnected(SizeType output_dim, const Parameters& config) :
  Base(0, output_dim),
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
        w_.insert(idx, jdx) =
          config_.weight_mean + random(idx, jdx) * config_.weight_std;
      }
    }
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
  ar & config_.weight_std;

  // Save weight matrix
  ar & w_;

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
  ar & config_.weight_std;

  // Save weight matrix
  ar & w_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
