/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_RECEPTIVE_VOLUME_HPP
#define FFNN_LAYER_IMPL_RECEPTIVE_VOLUME_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
#define RECEPTIVE_VOLUME_TARGS ValueType,\
                               HeightAtCompileTime,\
                               WidthAtCompileTime,\
                               DepthAtCompileTime,\
                               FilterCountAtCompileTime,\
                               EmbeddingMode
#define RECEPTIVE_VOLUME ReceptiveVolume<RECEPTIVE_VOLUME_TARGS>

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
RECEPTIVE_VOLUME::
Parameters::Parameters(ScalarType init_weight_std,
                       ScalarType init_bias_std,
                       ScalarType init_weight_mean,
                       ScalarType init_bias_mean) :
  init_weight_std(init_weight_std),
  init_bias_std(init_bias_std),
  init_weight_mean(init_weight_mean),
  init_bias_mean(init_bias_mean)
{
  FFNN_ASSERT_MSG(init_bias_std > 0, "[init_bias_std] should be positive");
  FFNN_ASSERT_MSG(init_weight_std > 0, "[init_weight_std] should be positive");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
RECEPTIVE_VOLUME::ReceptiveVolume(const ShapeType& filter_shape, const SizeType& filter_count, const Parameters& config) :
  Base(filter_shape, ShapeType(1, 1, filter_count)),
  config_(config)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
RECEPTIVE_VOLUME::~ReceptiveVolume()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
bool RECEPTIVE_VOLUME::initialize()
{
  // Abort if layer is already initialized
  if (Base::setupRequired())
  {
    if (Base::isInitialized())
    {
      FFNN_WARN_NAMED("layer::ReceptiveVolume", "<" << Base::getID() << "> already initialized.");
      return false;
    }
    else
    {
      /// Reset all filters
      reset();
    }
  }

  // Set initialization flag
  Base::initialized_ = true;
  return Base::isInitialized();
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void RECEPTIVE_VOLUME::reset()
{
  // Deduce filter height
  const SizeType f_h =
    EmbeddingMode == ColEmbedding ?
    (Base::input_shape_.height * Base::input_shape_.depth) : Base::input_shape_.height;

  // Deduce filter width
  const SizeType f_w =
    EmbeddingMode == RowEmbedding ?
    (Base::input_shape_.width * Base::input_shape_.depth) : Base::input_shape_.width;

  // Resize the filter bank
  filter_bank_.resize(Base::output_shape_.depth);

  // Initializer all filters
  for (auto& filter : filter_bank_)
  {
    // Set uniformly random weight matrix + add biases
    filter.setRandom(f_h, f_w);
    filter *= config_.init_weight_std;
    if (std::abs(config_.init_weight_mean) > 0)
    {
      filter.array() += config_.init_weight_mean;
    }
  }

  // Set uniformly random bias matrix + add biases
  b_.setRandom(Base::output_shape_.depth, 1);
  b_ *= config_.init_bias_std;
  if (std::abs(config_.init_bias_mean) > 0)
  {
    b_.array() += config_.init_bias_mean;
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
template<typename InputBlockType, typename OutputBlockType>
void RECEPTIVE_VOLUME::forward(const Eigen::MatrixBase<InputBlockType>& input,
                               Eigen::MatrixBase<OutputBlockType> const& output)
{
  BiasVectorType out(Base::output_shape_.depth, 1);
  for (size_t idx = 0; idx < filter_bank_.size(); idx++)
  {
    // @see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
    out(idx) = (input.array() * filter_bank_[idx].array()).sum() + b_(idx);
  }

  const_cast<Eigen::MatrixBase<OutputBlockType>&>(output) += out + b_;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
template<typename InputBlockType, typename ForwardErrorBlockType>
void RECEPTIVE_VOLUME::backward(const Eigen::MatrixBase<InputBlockType>& input,
                                const Eigen::MatrixBase<ForwardErrorBlockType>& error)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void RECEPTIVE_VOLUME::save(typename RECEPTIVE_VOLUME::OutputArchive& ar,
                            typename RECEPTIVE_VOLUME::VersionType version) const
{
  ffnn::io::signature::apply<RECEPTIVE_VOLUME>(ar);
  Base::save(ar, version);

  // Save configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Save filters
  for (const auto& filter : filter_bank_)
  {
    ar & filter;
  }

  // Save bias matrix
  ar & b_;

  FFNN_DEBUG_NAMED("layer::ReceptiveVolume", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void RECEPTIVE_VOLUME::load(typename RECEPTIVE_VOLUME::InputArchive& ar,
                            typename RECEPTIVE_VOLUME::VersionType version)
{
  ffnn::io::signature::check<RECEPTIVE_VOLUME>(ar);
  Base::load(ar, version);

  // Load configuration parameters
  ar & config_.init_weight_std;
  ar & config_.init_weight_mean;
  ar & config_.init_bias_std;
  ar & config_.init_bias_mean;

  // Load filters
  filter_bank_.resize(Base::output_shape_.depth);
  for (SizeType idx = 0; idx < Base::output_shape_.depth; idx++)
  {
    ar & filter_bank_[idx];
  }

  // Load bias matrix
  ar & b_;

  FFNN_DEBUG_NAMED("layer::ReceptiveVolume", "Loaded");
}

#undef RECEPTIVE_VOLUME_TARGS
#undef RECEPTIVE_VOLUME
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_RECEPTIVE_VOLUME_HPP
