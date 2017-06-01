/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_CONVOLUTION_HPP
#define FFNN_LAYER_IMPL_CONVOLUTION_HPP

// C++ Standard Library
#include <exception>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/optimizer/none.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
template <typename ValueType,
          typename LayerTraits>
Convolution<ValueType, LayerTraits>::
Convolution(const Configuration& config) :
  Base(config.embedded_input_shape_, config.embedded_output_shape_)
{}

template <typename ValueType,
          typename LayerTraits>
Convolution<ValueType, LayerTraits>::~Convolution()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Convolution", "Destroying [layer::Convolution] object <" << this->getID() << ">");
}

template <typename ValueType,
          typename LayerTraits>
bool Convolution<ValueType, LayerTraits>::initialize()
{
  if (BaseType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Convolution", "<" << BaseType::getID() << "> already initialized.");
    return false;
  }
  else if (!BaseType::initialize())
  {
    FFNN_WARN_NAMED("layer::Convolution", "<" << BaseType::getID() << "> failed basic initializaition.");
    return false;
  }
  else if (BaseType::setupRequired())
  {
    reset();
  }

  // Setup optimizer
  if (static_cast<bool>(config_.optimizer_))
  {
    config_.optimizer_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::Convolution",
                   "<" <<
                   BaseType::getID() <<
                   "> initialized as (in=" <<
                   input_volume_shape_ <<
                   ", out=" <<
                   config_.output_shape_ <<
                   ") (depth_embedding=" <<
                   (Mode == ColEmbedding ? "Col" : "Row") <<
                   ", nfilters=" <<
                   config_.output_shape_.depth <<
                   ", stride=" <<
                   config_.stride_shape_ <<
                   ", optimizer=" <<
                   config_.optimizer_->name() <<
                   ")");
  return true;
}

template <typename ValueType,
          typename LayerTraits>
bool Convolution<ValueType, LayerTraits>::forward()
{
  // Run forward optimized iteration
  if (!config_.optimizer_->forward(*this))
  {
    return false;
  }

  // Perform convolution operation
  const auto& _is = config_.input_shape_;
  const auto& _os = config_.output_shape_;
  const auto& _ss = config_.stride_shape_;
  for (offset_type idx = 0, hdx = 0; idx < _os.height; idx++, hdx += _ss.height)
  {
    for (offset_type jdx = 0, wdx = 0; jdx < _os.width; jdx++, wdx += _ss.width)
    {
      offset_type kdx = 0;
      for (const auto& kernel : parameters_)
      {
        output_mappings_[idx][jdx][kdx++] = 
          kernel.cwiseProduct(BaseType::input_.block(hdx, wdx, _is.height, _is.width)) +
          parameters_.bias;
      }
    }
  }
  return true;
}

template <typename ValueType,
          typename LayerTraits>
bool Convolution<ValueType, LayerTraits>::backward()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");

  // Reset backward error values
  BaseType::backward_error_.setZero();

  // Backprop error
  const auto& _is = config_.input_shape_;
  const auto& _os = config_.output_shape_;
  const auto& _ss = config_.stride_shape_;
  for (offset_type idx = 0, hdx = 0; idx < _os.height; idx++, hdx += _ss.height)
  {
    for (offset_type jdx = 0, wdx = 0; jdx < _os.width; jdx++, wdx += _ss.width)
    {
      // Sum over all filters
      offset_type kdx = 0;
      for (const auto& kernel : parameters_)
      {
        BaseType::backward_error_.block(hdx, wdx, _is.height, _is.width) +=
          kernel * forward_error_mappings_[idx][jdx][kdx++];
      }
    }
  }

  // Run optimizer
  return config_.optimizer_->backward(*this);
}

template <typename ValueType,
          typename LayerTraits>
bool Convolution<ValueType, LayerTraits>::update()
{
  FFNN_ASSERT_MSG(config_.optimizer_, "No optimization resource set.");
  return config_.optimizer_->update(*this);
}

template <typename ValueType,
          typename LayerTraits>
void Convolution<ValueType, LayerTraits>::reset()
{
  // Create receptive field
  const auto& _is = config_.input_shape_;
  const auto& _os = config_.output_shape_;
  const auto& _fs = config_.filter_shape_;
  parameters_.setZeros(_fs.height, _fs.width, _is.depth, _os.depth);
}

template <typename ValueType,
          typename LayerTraits>
void Convolution<ValueType, LayerTraits>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  config_.optimizer_ = opt;
}

template <typename ValueType,
          typename LayerTraits>
offset_type Convolution<ValueType, LayerTraits>::connectToForwardLayer(const Layer<ValueType>& next, offset_type offset)
{
  // Connect outputs
  const offset_type data_offset = BaseType::connectToForwardLayer(next, offset);

  // Resize mapping matrices
  const auto& _os = config_.output_shape_;
  forward_error_mappings_.resize(boost::extents[_os.height][_os.width]);
  output_mappings_.resize(boost::extents[_os.height][_os.width]);

  // Map outputs and forward error values
  ValueType* out_ptr = const_cast<ValueType*>(next.getInputBuffer().data());
  ValueType* err_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
  for (size_type idx = 0; idx < _os.height; idx++)
  {
    for (size_type jdx = 0; jdx < _os.width; jdx++)
    {
      // Pointer offset
      const offset_type kdx = (LayerTraits::embedding_mode == ColEmbedding) ?
                              jdx * BaseType::output_shape_.height + idx * _os.depth :
                              idx * BaseType::output_shape_.width  + jdx * _os.depth;

      // Set backward-error memory mapping
      forward_error_mappings_[idx][jdx] = err_ptr + kdx;

      // Set output memory mapping
      output_mappings_[idx][jdx] = out_ptr + kdx;
    }
  }
  return data_offset;
}

template <typename ValueType,
          typename LayerTraits>
void Convolution<ValueType, LayerTraits>::save(OutputArchive& ar, VersionType version) const
{
  ffnn::io::signature::apply<SelfType>(ar);
  BaseType::save(ar, version);

  // Save volumes
  ar & parameters_;

  FFNN_DEBUG_NAMED("layer::Convolution", "Saved");
}

template <typename ValueType,
          typename LayerTraits>
void Convolution<ValueType, LayerTraits>::load(InputArchive& ar, VersionType version)
{
  ffnn::io::signature::check<SelfType>(ar);
  BaseType::load(ar, version);

  // Load volumes
  ar & parameters_;

  FFNN_DEBUG_NAMED("layer::Convolution", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_CONVOLUTION_HPP
