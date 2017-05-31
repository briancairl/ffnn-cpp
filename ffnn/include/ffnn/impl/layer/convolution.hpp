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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
Convolution<TARGS>::
Convolution(const Configuration& config) :
  Base(config.embedded_input_shape_, config.embedded_output_shape_)
  Base(ShapeType(convolution::embed_dimension<Mode, ColEmbedding>(input_shape.height, input_shape.depth),
                 convolution::embed_dimension<Mode, RowEmbedding>(input_shape.width, input_shape.depth)),
       ShapeType(convolution::embed_dimension<Mode, ColEmbedding>(convolution::output_dimension(input_shape.height, filter_shape.height, stride), filter_shape.depth),
                 convolution::embed_dimension<Mode, RowEmbedding>(convolution::output_dimension(input_shape.width, filter_shape.width, stride), filter_shape.depth))),
  input_volume_shape_(input_shape),
  output_volume_shape_(convolution::output_dimension(input_shape.height, filter_shape.height, stride),
                       convolution::output_dimension(input_shape.width, filter_shape.width, stride),
                       filter_shape.depth),
  filter_shape_(convolution::embed_dimension<Mode, ColEmbedding>(filter_shape.height, input_shape.depth),
                convolution::embed_dimension<Mode, RowEmbedding>(filter_shape.width,  input_shape.depth)),
  stride_(convolution::embed_dimension<Mode, ColEmbedding>(stride, input_shape.depth),
          convolution::embed_dimension<Mode, RowEmbedding>(stride, input_shape.depth))
{}

template<typename ValueType,
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
Convolution<TARGS>::~Convolution()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Convolution", "Destroying [layer::Convolution] object <" << this->getID() << ">");
}

template<typename ValueType,
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
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
  for (OffsetType idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += stride_.height)
  {
    for (OffsetType jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += stride_.width)
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");

  BaseType::backward_error_.setZero();

  // Get block dimensions
  const auto& ris = parameters_.getInputShape();

  // Compute outputs through volumes
  for (OffsetType idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += stride_.height)
  {
    for (OffsetType jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += stride_.width)
    {
      // Sum over all filters
      OffsetType kdx = 0;
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool Convolution<TARGS>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::reset()
{
  // Create receptive field
  new (&parameters_) ParametersType(filter_shape_, output_volume_shape_.depth);
}

template<typename ValueType,
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void Convolution<TARGS>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
typename
Convolution<TARGS>::OffsetType
Convolution<TARGS>::connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset)
{
  // Connect outputs
  OffsetType offset_after_connect = BaseType::connectToForwardLayer(next, offset);

  // Resize mapping matrices
  output_mappings_.resize(boost::extents[output_volume_shape_.height][output_volume_shape_.width]);
  forward_error_mappings_.resize(boost::extents[output_volume_shape_.height][output_volume_shape_.width]);

  // Map to individual volumes
  ValueType* out_ptr = const_cast<ValueType*>(next.getInputBuffer().data());
  ValueType* err_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
  for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
    {
      // Compute pointer offset
      const OffsetType kdx = (Mode == ColEmbedding) ?
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
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
         ffnn::size_type HeightAtCompileTime,
         ffnn::size_type WidthAtCompileTime,
         ffnn::size_type DepthAtCompileTime,
         ffnn::size_type FilterHeightAtCompileTime,
         ffnn::size_type FilterWidthAtCompileTime,
         ffnn::size_type FilterDepthAtCompileTime,
         ffnn::size_type StrideAtCompileTime,
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
