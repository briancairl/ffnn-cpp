/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_LOCAL_CONVOLUTION_HPP
#define FFNN_LAYER_IMPL_LOCAL_CONVOLUTION_HPP

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
template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
LocalConvolution<TARGS>::
LocalConvolution(const ShapeType& input_shape,
                 const SizeType& filter_height,
                 const SizeType& filter_width,
                 const SizeType& filter_count,
                 const SizeType& filter_stride) :
  Base(ShapeType(embed_dimension<Mode, ColEmbedding>(input_shape.height, input_shape.depth),
                 embed_dimension<Mode, RowEmbedding>(input_shape.width, input_shape.depth)),
       ShapeType(embed_dimension<Mode, ColEmbedding>(output_dimension(input_shape.height, filter_height, filter_stride), filter_count),
                 embed_dimension<Mode, RowEmbedding>(output_dimension(input_shape.width,  filter_width,  filter_stride), filter_count))),
  input_volume_shape_(input_shape),
  output_volume_shape_(output_dimension(input_shape.height,  filter_height,  filter_stride),
                       output_dimension(input_shape.width,  filter_width,  filter_stride),
                       filter_count),
  filter_shape_(embed_dimension<Mode, ColEmbedding>(filter_height, input_shape.depth),
                embed_dimension<Mode, RowEmbedding>(filter_width,  input_shape.depth)),
  filter_stride_(embed_dimension<Mode, ColEmbedding>(filter_stride, input_shape.depth),
                 embed_dimension<Mode, RowEmbedding>(filter_stride, input_shape.depth)),
  opt_(boost::make_shared<typename optimizer::None<Self>>())
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
LocalConvolution<TARGS>::~LocalConvolution()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::LocalConvolution", "Destroying [layer::LocalConvolution] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool LocalConvolution<TARGS>::initialize()
{
  if (Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::LocalConvolution", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    FFNN_WARN_NAMED("layer::LocalConvolution", "<" << Base::getID() << "> failed basic initializaition.");
    return false;
  }
  else if (Base::setupRequired())
  {
    reset();
  }

  // Setup optimizer
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::LocalConvolution",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   input_volume_shape_ <<
                   ", out=" <<
                   output_volume_shape_ <<
                   ") (depth_embedding=" <<
                   (Mode == ColEmbedding ? "Col" : "Row") <<
                   ", nfilters=" <<
                   output_volume_shape_.depth <<
                   ", stride=" <<
                   filter_stride_ <<
                   ", optimizer=" <<
                   opt_->name() <<
                   ")");
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
template<typename WeightDistribution,
         typename BiasDistribution>
bool  LocalConvolution<TARGS>::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
{
  // Abort if layer is already initialized
  if (!Base::setupRequired())
  {
    throw std::logic_error("Wrong initialization method called. This is a loaded object.");
  }
  else if (!initialize())
  {
    return false;
  }

  // Intiialize all volumes
  for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
    {
      // Initialize field
      Base::initialized_ &= parameters_[idx][jdx].initialize(wd, bd);

      // Check that last created field was initialized properly
      FFNN_ASSERT_MSG(Base::initialized_, "Failed to initialize receptor.");
    }
  }
  return Base::initialized_;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool LocalConvolution<TARGS>::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute outputs through volumes
  for (OffsetType idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += filter_stride_.height)
  {
    for (OffsetType jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += filter_stride_.width)
    {
      // Get block dimensions
      const auto& ris = parameters_[idx][jdx].inputShape();

      // Activate receptor
      parameters_[idx][jdx].forward(Base::input_.block(hdx, wdx, ris.height, ris.width));
    }
  }
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool LocalConvolution<TARGS>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");

  Base::backward_error_.setZero();

  // Compute outputs through volumes
  for (OffsetType idx = 0, hdx = 0; idx < output_volume_shape_.height; idx++, hdx += filter_stride_.height)
  {
    for (OffsetType jdx = 0, wdx = 0; jdx < output_volume_shape_.width; jdx++, wdx += filter_stride_.width)
    {
      // Get block dimensions
      const auto& ris = parameters_[idx][jdx].inputShape();

      // Sum over all filters
      OffsetType kdx = 0;
      const ValueType* err = parameters_[idx][jdx].getForwardErrorMapping();
      for (const auto& filter : parameters_[idx][jdx].filters)
      {
        Base::backward_error_.block(hdx, wdx, ris.height, ris.width) += filter.kernel * err[kdx++];
      }
    }
  }

  // Run optimizer
  return opt_->backward(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
bool LocalConvolution<TARGS>::update()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
  return opt_->update(*this);
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void LocalConvolution<TARGS>::reset()
{
  parameters_.resize(boost::extents[output_volume_shape_.height][output_volume_shape_.width]);
  for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
  {
    for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
    {
      // Create receptive field
      new (&parameters_[idx][jdx]) ParameterBlockType(filter_shape_, output_volume_shape_.depth);
    }
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void LocalConvolution<TARGS>::setOptimizer(typename Optimizer::Ptr opt)
{
  FFNN_ASSERT_MSG(opt, "Input optimizer object is an empty resource.");
  opt_ = opt;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
typename
LocalConvolution<TARGS>::OffsetType
LocalConvolution<TARGS>::connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset)
{
  // Connect outputs
  OffsetType offset_after_connect = Base::connectToForwardLayer(next, offset);

  // Map to individual volumes
  ValueType* output_ptr = const_cast<ValueType*>(next.getInputBuffer().data());
  ValueType* error_ptr  = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
  for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
  {
    for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
    {
      // Compute pointer offset
      const OffsetType kdx =
        jdx * ((Mode == ColEmbedding) ? Base::output_shape_.height : Base::output_shape_.width) +
        idx * output_volume_shape_.depth;

      // Set output memory mapping
      parameters_[idx][jdx].setOutputMapping(output_ptr + kdx);

      // Set backward-error memory mapping
      parameters_[idx][jdx].setForwardErrorMapping(error_ptr + kdx);
    }
  }
  return offset_after_connect;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void LocalConvolution<TARGS>::save(typename LocalConvolution<TARGS>::OutputArchive& ar,
                                   typename LocalConvolution<TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<LocalConvolution<TARGS>>(ar);
  Base::save(ar, version);

  // Save volumes
  for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
    {
      parameters_[idx][jdx].save(ar, version);
    }
  }
  FFNN_DEBUG_NAMED("layer::LocalConvolution", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         EmbeddingMode Mode,
         typename _HiddenLayerShape>
void LocalConvolution<TARGS>::load(typename LocalConvolution<TARGS>::InputArchive& ar,
                                   typename LocalConvolution<TARGS>::VersionType version)
{
  ffnn::io::signature::check<LocalConvolution<TARGS>>(ar);
  Base::load(ar, version);

  // Load volumes
  reset();
  for (SizeType idx = 0; idx < output_volume_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < output_volume_shape_.width; jdx++)
    {
      parameters_[idx][jdx].load(ar, version);
    }
  }

  FFNN_DEBUG_NAMED("layer::LocalConvolution", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_LOCAL_CONVOLUTION_HPP