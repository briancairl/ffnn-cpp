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
#define FULLY_CONNECTED_TARGS ValueType,\
                              InputsAtCompileTime,\
                              OutputsAtCompileTime,\
                              _HiddenLayerShape
#define FULLY_CONNECTED FullyConnected<FULLY_CONNECTED_TARGS>

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
FULLY_CONNECTED::FullyConnected(SizeType output_size) :
  Base(ShapeType(InputsAtCompileTime), ShapeType(output_size)),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
FULLY_CONNECTED::~FullyConnected()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected", "Destroying [layer::FullyConnected] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool FULLY_CONNECTED::initialize()
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
         typename BiasDistribution>
bool FULLY_CONNECTED::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
{
  if (initialize())
  {
    if (Base::setupRequired())
    {
      // Set layer connections weights
      {
        auto coeffInitfn = [](ValueType x, const WeightDistribution& dist)
        {
          return dist.generate();
        };
        w_ = w_.unaryExpr(boost::bind<ValueType>(coeffInitfn, _1, wd));
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
    FFNN_WARN_NAMED("layer::FullyConnected",
                    "Layer was previously loaded. Trained parameters will not be reset.");
    return false;
  }
  return false;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool FULLY_CONNECTED::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute weighted + biased outputs
  Base::output_.noalias() = w_ * Base::input_;
  Base::output_ += b_;
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool FULLY_CONNECTED::backward()
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
bool FULLY_CONNECTED::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FULLY_CONNECTED::reset()
{
  // Zero out connection weights and biases with appropriate size
  w_.setZero(Base::outputShape().size(), Base::inputShape().size());
  b_.setZero(Base::outputShape().size(), 1);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FULLY_CONNECTED::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FULLY_CONNECTED::save(typename FULLY_CONNECTED::OutputArchive& ar,
                           typename FULLY_CONNECTED::VersionType version) const
{
  ffnn::io::signature::apply<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
  Base::save(ar, version);

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FULLY_CONNECTED::load(typename FULLY_CONNECTED::InputArchive& ar,
                           typename FULLY_CONNECTED::VersionType version)
{
  ffnn::io::signature::check<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
  Base::load(ar, version);

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
