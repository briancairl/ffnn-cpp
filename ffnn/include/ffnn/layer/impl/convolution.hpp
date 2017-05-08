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
Convolution<CONVOLUTION_TARGS>::
Convolution(const ShapeType& input_shape,
            const SizeType& filter_height,
            const SizeType& filter_width,
            const SizeType& filter_count,
            const SizeType& filter_stride,
            const Parameters& config) :
  Base(input_shape,
       ShapeType(RESOLVE_CONVOLUTION_OUTPUT(input_shape.height, filter_height, filter_stride),
                 RESOLVE_CONVOLUTION_OUTPUT(input_shape.width, filter_width, filter_stride),
                 filter_count)),
  config_(config),
  filter_shape_(filter_height, filter_width, input_shape.depth),
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
Convolution<CONVOLUTION_TARGS>::~Convolution()
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
bool Convolution<CONVOLUTION_TARGS>::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Colvolution", "<" << Base::getID() << "> already initialized.");
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

  FFNN_DEBUG_NAMED("layer::Colvolution",
                   "<" <<
                   Base::getID() <<
                   "> initialized as (in=" <<
                   Base::inputShape() <<
                   ", out=" <<
                   Base::outputShape() <<
                   ") [nfilters= " <<
                   filter_count_ <<
                   "stride= " <<
                   filter_stride_ <<
                   " (optimizer=" <<
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
bool Convolution<CONVOLUTION_TARGS>::forward()
{
  if (!opt_->forward(*this))
  {
    return false;
  }

  // Compute outputs through volumes
  for (OffsetType idx = 0, sidx = 0; idx < Base::output_shape_.height; idx++, sidx += filter_stride_)
  {
    OffsetType kdx = 0;
    for (OffsetType jdx = 0, sjdx = 0; jdx < Base::output_shape_.width; jdx++, sjdx += filter_stride_)
    {
      // Get block dimensions
      const auto& ris = receptors_[idx][jdx]->inputShape();
      const auto& ros = receptors_[idx][jdx]->outputShape();

      // Activate receptor
      receptors_[idx][jdx]->forward(
        Base::input_->block(ris.height, ris.width, sidx, sjdx),
        Base::output_->block(ros.depth, 1, kdx, jdx)
      );
      kdx += ros.depth;
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
         FFNN_SIZE_TYPE EmbeddingMode>
bool Convolution<CONVOLUTION_TARGS>::backward()
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
bool Convolution<CONVOLUTION_TARGS>::update()
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
void Convolution<CONVOLUTION_TARGS>::reset()
{
  receptors_.resize(boost::extents[Base::output_shape_.height][Base::output_shape_.width]);
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      receptors_[idx][jdx] = boost::make_shared<ConvolutionVolumeType>(filter_shape_, filter_count_);
      Base::initialized_  &= receptors_[idx][jdx]->initialize(config_);

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
bool Convolution<CONVOLUTION_TARGS>::computeBackwardError()
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
void Convolution<CONVOLUTION_TARGS>::
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
void Convolution<CONVOLUTION_TARGS>::
  save(typename Convolution<CONVOLUTION_TARGS>::OutputArchive& ar,
       typename Convolution<CONVOLUTION_TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<Convolution<CONVOLUTION_TARGS>>(ar);
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save volumes
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      receptors_[idx][jdx]->save(ar, version);
    }
  }

  FFNN_DEBUG_NAMED("layer::Colvolution", "Saved");
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
void Convolution<CONVOLUTION_TARGS>::
  load(typename Convolution<CONVOLUTION_TARGS>::InputArchive& ar,
       typename Convolution<CONVOLUTION_TARGS>::VersionType version)
{
  ffnn::io::signature::check<Convolution<CONVOLUTION_TARGS>>(ar);
  Base::load(ar, version);

  // Load configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Load volumes
  reset();
  for (SizeType idx = 0; idx < Base::output_shape_.height; idx++)
  {
    for (SizeType jdx = 0; jdx < Base::output_shape_.width; jdx++)
    {
      receptors_[idx][jdx]->load(ar, version);
    }
  }

  FFNN_DEBUG_NAMED("layer::Colvolution", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
