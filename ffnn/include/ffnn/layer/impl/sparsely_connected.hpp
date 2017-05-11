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
#define SPARSELY_CONNECTED_TARGS ValueType, InputsAtCompileTime, OutputsAtCompileTime
#define SPARSELY_CONNECTED_TARGS_ADVANCED _HiddenLayerShape
#define SPARSELY_CONNECTED SparselyConnected<SPARSELY_CONNECTED_TARGS, SPARSELY_CONNECTED_TARGS_ADVANCED>

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
SPARSELY_CONNECTED::
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
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
SPARSELY_CONNECTED::
SparselyConnected(SizeType output_size, const Parameters& config) :
  Base(ShapeType(0), ShapeType(output_size)),
  config_(config),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
SPARSELY_CONNECTED::~SparselyConnected()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool SPARSELY_CONNECTED::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::SparselyConnected", "<" << Base::getID() << "> already initialized.");
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

  FFNN_DEBUG_NAMED("layer::SparselyConnected",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::inputShape().size() <<
                   ", out=" <<
                   Base::outputShape().size() <<
                   ") [with 1 biasing input] (optimizer=" <<
                   opt_->name() <<
                   ")");
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool SPARSELY_CONNECTED::forward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute weighted outputs
  Base::output_.noalias() = w_ * Base::input_;
  Base::output_.noalias() += b_;
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool SPARSELY_CONNECTED::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");

  // Compute backward error
  Base::backward_error_.noalias() = w_.transpose() * Base::forward_error_;

  // Run optimizer
  return opt_->backward(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool SPARSELY_CONNECTED::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void SPARSELY_CONNECTED::reset()
{
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime> RandMatrix;

  FFNN_ASSERT_MSG(Base::isInitialized(), "Layer is not initialized.");

  // Create random value matrix
  RandMatrix random(Base::outputShape().size(), Base::inputShape().size());
  random.setRandom(Base::outputShape().size(), Base::inputShape().size());

  // Build weight matrix
  w_.resize(Base::outputShape().size(), Base::inputShape().size());
  for (SizeType idx = 0; idx < Base::outputShape().size(); idx++)
  {
    for (SizeType jdx = 0; jdx < Base::inputShape().size(); jdx++)
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
  b_.setRandom(Base::outputShape().size(), 1);
  b_ *= config_.init_bias_std;
  if (std::abs(config_.init_bias_mean) > 0)
  {
    b_.array() += config_.init_bias_mean;
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void SPARSELY_CONNECTED::prune(ValueType epsilon)
{
  w_.prune(0, epsilon);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void SPARSELY_CONNECTED::
  setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void SPARSELY_CONNECTED::
  save(typename SPARSELY_CONNECTED::OutputArchive& ar,
       typename SPARSELY_CONNECTED::VersionType version) const
{
  ffnn::io::signature::apply<SPARSELY_CONNECTED>(ar);
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
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void SPARSELY_CONNECTED::
  load(typename SPARSELY_CONNECTED::InputArchive& ar,
       typename SPARSELY_CONNECTED::VersionType version)
{
  ffnn::io::signature::check<SPARSELY_CONNECTED>(ar);
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
