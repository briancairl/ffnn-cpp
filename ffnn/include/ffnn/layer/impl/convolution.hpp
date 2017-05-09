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
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
Convolution<CONV_TARGS>::
Convolution(const ShapeType& input_shape,
            const SizeType& filter_height,
            const SizeType& filter_width,
            const SizeType& filter_count,
            const SizeType& filter_stride) :
  Base(ShapeType(CONV_EMBEDDED_H(input_shape.height, input_shape.depth),
                 CONV_EMBEDDED_W(input_shape.width, input_shape.depth)),
       ShapeType(CONV_EMBEDDED_H(CONV_LENGTH_WITH_STRIDE(input_shape.height, filter_height, filter_stride), filter_count),
                 CONV_EMBEDDED_W(CONV_LENGTH_WITH_STRIDE(input_shape.width,  filter_width,  filter_stride), filter_count))),
  filter_shape_(CONV_EMBEDDED_H(filter_height, input_shape.depth),
                CONV_EMBEDDED_W(filter_width,  input_shape.depth)),
  filter_count_(filter_count),
  filter_stride_(filter_stride),
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
         FFNN_SIZE_TYPE EmbeddingMode>
Convolution<CONV_TARGS>::~Convolution()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Convolution", "Destroying [layer::Convolution] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::initialize()
{
  return initialize(Convolution<CONV_TARGS>::Parameters());
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::initialize(const Convolution<CONV_TARGS>::Parameters& config)
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Convolution", "<" << Base::getID() << "> already initialized.");
    return false;
  }
  else if (!Base::initialize())
  {
    return false;
  }

  // Initialize weights
  if (Base::setupRequired())
  {
    reset(config);
  }

  // Setup optimizer
  if (opt_)
  {
    opt_->initialize(*this);
  }

  FFNN_DEBUG_NAMED("layer::Convolution",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::inputShape() <<
                   ", out=" <<
                   Base::outputShape() <<
                   ") (depth_embedding=" <<
                   (EmbeddingMode == ColEmbedding ? "Col" : "Row") <<
                   ", nfilters=" <<
                   filter_count_ <<
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
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute outputs through volumes
  OffsetType kdx = 0;
  for (OffsetType idx = 0, idx_str = 0; idx < Base::output_shape_.height; idx++, idx_str += filter_stride_)
  {
    for (OffsetType jdx = 0, jdx_str = 0; jdx < Base::output_shape_.width; jdx++, jdx_str += filter_stride_)
    {
      // Get block dimensions
      const auto& ris = receptors_[idx][jdx].inputShape();
      const auto& ros = receptors_[idx][jdx].outputShape();

      // Activate receptor
      receptors_[idx][jdx].forward(
        Base::input_.block(idx_str, jdx_str, ris.height, ris.width),
        Base::output_.block(kdx, jdx, ros.depth, 1)
      );
    }
    kdx += filter_count_;
  }
  FFNN_INFO("BLOOP");
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
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::backward()
{
  FFNN_ASSERT_MSG(opt_, "No optimization resource set.");
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
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::update()
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
         FFNN_SIZE_TYPE EmbeddingMode>
void Convolution<CONV_TARGS>::reset(const Convolution<CONV_TARGS>::Parameters& config)
{
  receptors_.resize(boost::extents[Base::output_shape_.height][Base::output_shape_.width]);
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      // Create receptive field
      new (&receptors_[idx][jdx]) ConvolutionVolumeType(filter_shape_, filter_count_);

      // Initialize field
      Base::initialized_ &= receptors_[idx][jdx].initialize(config);

      // Check that last created field was initialized properly
      FFNN_ASSERT_MSG(Base::initialized_, "Failed to initialize receptor.");
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
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONV_TARGS>::computeBackwardError()
{
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
         FFNN_SIZE_TYPE EmbeddingMode>
void Convolution<CONV_TARGS>::
  setOptimizer(typename Optimizer::Ptr opt)
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
         FFNN_SIZE_TYPE EmbeddingMode>
void Convolution<CONV_TARGS>::
  save(typename Convolution<CONV_TARGS>::OutputArchive& ar,
       typename Convolution<CONV_TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<Convolution<CONV_TARGS>>(ar);
  Base::save(ar, version);

  // Save volumes
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      receptors_[idx][jdx].save(ar, version);
    }
  }

  FFNN_DEBUG_NAMED("layer::Convolution", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterHeightAtCompileTime,
         FFNN_SIZE_TYPE FilterWidthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE StrideAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void Convolution<CONV_TARGS>::
  load(typename Convolution<CONV_TARGS>::InputArchive& ar,
       typename Convolution<CONV_TARGS>::VersionType version)
{
  ffnn::io::signature::check<Convolution<CONV_TARGS>>(ar);
  Base::load(ar, version);

  // Load volumes
  reset();
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      receptors_[idx][jdx].load(ar, version);
    }
  }

  FFNN_DEBUG_NAMED("layer::Convolution", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
