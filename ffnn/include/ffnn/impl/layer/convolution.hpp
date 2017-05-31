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

#define TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterDepthAtCompileTime,\
  StrideAtCompileTime,\
  Mode,\
  _HiddenLayerShape

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
Convolution<TARGS>::
Convolution(const Configuration& config) :
  Base(config.embedded_input_shape_, config.embedded_output_shape_)
{}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
Convolution<TARGS>::~Convolution()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Convolution", "Destroying [layer::Convolution] object <" << this->getID() << ">");
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::initialize()
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
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::Convolution",
                   "<" <<
                   BaseType::getID() <<
                   "> initialized as (in=" <<
                   input_volume_shape_ <<
                   ", out=" <<
                   output_volume_shape_ <<
                   ") (depth_embedding=" <<
                   (Mode == ColEmbedding ? "Col" : "Row") <<
                   ", nfilters=" <<
                   output_volume_shape_.depth <<
                   ", stride=" <<
                   stride_ <<
                   ", optimizer=" <<
                   opt_->name() <<
                   ")");
  return true;
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
template<typename WeightDistribution,
         typename BiasDistribution>
bool  Convolution<TARGS>::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
{
  // Abort if layer is already initialized
  if (!BaseType::setupRequired())
  {
    throw std::logic_error("Wrong initialization method called. This is a loaded object.");
  }
  else if (!initialize())
  {
    return false;
  }

  // Intiialize shared weights
  BaseType::initialized_ &= parameters_.initialize(wd, bd);
  FFNN_ASSERT_MSG(BaseType::initialized_, "Failed to initialize receptor.");

  return BaseType::initialized_;
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Get block dimensions
  const auto& ris = parameters_.getInputShape();

  // Compute outputs through volumes
  for (offset_type idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += stride_.height)
  {
    for (offset_type jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += stride_.width)
    {
      // Set output pointer
      parameters_.setOutputMapping(output_mappings_[idx][jdx]);

      // Activate receptor
      parameters_.forward(BaseType::input_.block(hdx, wdx, ris.height, ris.width));
    }
  }
  return true;
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");

  BaseType::backward_error_.setZero();

  // Get block dimensions
  const auto& ris = parameters_.getInputShape();

  // Compute outputs through volumes
  for (offset_type idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += stride_.height)
  {
    for (offset_type jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += stride_.width)
    {
      // Sum over all filters
      offset_type kdx = 0;
      for (const auto& filter : parameters_.filters)
      {
        BaseType::backward_error_.block(hdx, wdx, ris.height, ris.width) +=
          filter.kernel * forward_error_mappings_[idx][jdx][kdx++];
      }
    }
  }

  // Run optimizer
  return opt_->backward(*this);
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::reset()
{
  // Create receptive field
  new (&parameters_) ParametersType(filter_shape_, output_volume_shape_.depth);
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
offset_type Convolution<TARGS>::connectToForwardLayer(const Layer<ValueType>& next, offset_type offset)
{
  // Connect outputs
  offset_type offset_after_connect = BaseType::connectToForwardLayer(next, offset);

  // Resize mapping matrices
  output_mappings_.resize(boost::extents[output_volume_shape_.height][output_volume_shape_.width]);
  forward_error_mappings_.resize(boost::extents[output_volume_shape_.height][output_volume_shape_.width]);

  // Map to individual volumes
  ValueType* out_ptr = const_cast<ValueType*>(next.getInputBuffer().data());
  ValueType* err_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
  for (size_type idx = 0; idx < output_volume_shape_.height; idx++)
  {
    for (size_type jdx = 0; jdx < output_volume_shape_.width; jdx++)
    {
      // Compute pointer offset
      const offset_type kdx = (Mode == ColEmbedding) ?
                             jdx * BaseType::output_shape_.height + idx * output_volume_shape_.depth :
                             idx * BaseType::output_shape_.width  + jdx * output_volume_shape_.depth;

      // Set output memory mapping
      output_mappings_[idx][jdx] = out_ptr + kdx;

      // Set backward-error memory mapping
      forward_error_mappings_[idx][jdx] = err_ptr + kdx;
    }
  }
  return offset_after_connect;
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::save(typename Convolution<TARGS>::OutputArchive& ar,
                              typename Convolution<TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<SelfType>(ar);
  BaseType::save(ar, version);

  // Save volumes
  parameters_.save(ar, version);

  FFNN_DEBUG_NAMED("layer::Convolution", "Saved");
}

template<typename ValueType,
         size_type HeightAtCompileTime,
         size_type WidthAtCompileTime,
         size_type DepthAtCompileTime,
         size_type FilterHeightAtCompileTime,
         size_type FilterWidthAtCompileTime,
         size_type FilterDepthAtCompileTime,
         size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::load(typename Convolution<TARGS>::InputArchive& ar,
                              typename Convolution<TARGS>::VersionType version)
{
  ffnn::io::signature::check<SelfType>(ar);
  BaseType::load(ar, version);

  // Load volumes
  reset();
  parameters_.load(ar, version);

  FFNN_DEBUG_NAMED("layer::Convolution", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_CONVOLUTION_HPP
