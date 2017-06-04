/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP
#define FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP

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
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
FullyConnected<TARGS>::FullyConnected(SizeType output_size) :
  Base(ShapeType(InputsAtCompileTime), ShapeType(output_size)),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
FullyConnected<TARGS>::~FullyConnected()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected", "Destroying [layer::FullyConnected] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
bool FullyConnected<TARGS>::initialize()
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
                   Base::getInputShape().size() <<
                   ", out=" <<
                   Base::getOutputShape().size() <<
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
bool FullyConnected<TARGS>::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
{
  if (initialize())
  {
    if (Base::setupRequired())
    {
      // Set layer connections weights
      {
        auto fn = [](ValueType x, const WeightDistribution& dist) {return dist.generate();};
        w_ = w_.unaryExpr(boost::bind<ValueType>(fn, _1, wd));
      }

      // Set layer biases
      {
        auto fn = [](ValueType x, const BiasDistribution& dist) {return dist.generate();};
        b_ = b_.unaryExpr(boost::bind<ValueType>(fn, _1, bd));
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
bool FullyConnected<TARGS>::forward()
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
bool FullyConnected<TARGS>::backward()
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
bool FullyConnected<TARGS>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FullyConnected<TARGS>::reset()
{
  // Zero out connection weights and biases with appropriate size
  w_.setZero(Base::getOutputShape().size(), Base::getInputShape().size());
  b_.setZero(Base::getOutputShape().size(), 1);
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FullyConnected<TARGS>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _HiddenLayerShape>
void FullyConnected<TARGS>::save(typename FullyConnected<TARGS>::OutputArchive& ar,
                                 typename FullyConnected<TARGS>::VersionType version) const
{
  ffnn::internal::signature::apply<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
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
void FullyConnected<TARGS>::load(typename FullyConnected<TARGS>::InputArchive& ar,
                                 typename FullyConnected<TARGS>::VersionType version)
{
  ffnn::internal::signature::check<FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _HiddenLayerShape>>(ar);
  Base::load(ar, version);

  // Save weight/bias matrix
  ar & w_;
  ar & b_;

  FFNN_DEBUG_NAMED("layer::FullyConnected", "Loaded");
}

}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_FULLY_CONNECTED_HPP
