/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// Boost
#include <boost/bind.hpp>

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
SparselyConnected(SizeType output_size) :
  Base(ShapeType(0), ShapeType(output_size)),
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
template<typename WeightDistribution,
         typename BiasDistribution,
         typename ConnectionDistribution>
bool SPARSELY_CONNECTED::initialize(const WeightDistribution& wd,
                                    const BiasDistribution& bd,
                                    const ConnectionDistribution& cd,
                                    ValueType connection_probability)
{
  if (initialize())
  {
    if (Base::setupRequired())
    {
      // Build weight matrix
      for (SizeType idx = 0; idx < Base::outputShape().size(); idx++)
      {
        for (SizeType jdx = 0; jdx < Base::inputShape().size(); jdx++)
        {
          if (cd.cdf(cd.generate()) < connection_probability)
          {
            w_.insert(idx, jdx) = wd.generate();
          }
        }
      }

      // Set layer biases
      {
        auto coeffInitfn = [](ValueType x, const BiasDistribution& dist)
        {
          return dist.generate();
        };
        b_ = b_.unaryExpr(boost::bind<ValueType>(coeffInitfn, _1, bd));
      }
      return true;
    }
    FFNN_WARN_NAMED("layer::SparselyConnected",
                    "Layer was previously loaded. Trained parameters will not be reset.");
    return false;
  }
  return false;
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
  Base::output_ += b_;
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
  // Zero out connection weights and biases with appropriate size
  w_.resize(Base::outputShape().size(), Base::inputShape().size());
  b_.setRandom(Base::outputShape().size(), 1);
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

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::SparselyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
