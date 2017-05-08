/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP
#define FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
#define CONVOLUTION_VOLUME_TARGS ValueType,\
                               HeightAtCompileTime,\
                               WidthAtCompileTime,\
                               DepthAtCompileTime,\
                               FilterCountAtCompileTime,\
                               EmbeddingMode
#define CONVOLUTION_VOLUME ConvolutionVolume<CONVOLUTION_VOLUME_TARGS>

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
CONVOLUTION_VOLUME::
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
CONVOLUTION_VOLUME::ConvolutionVolume(const ShapeType& filter_shape, const SizeType& filter_count) :
  Base(filter_shape, ShapeType(1, 1, filter_count)),
  bank_(filter_count)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
CONVOLUTION_VOLUME::~ConvolutionVolume()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
bool CONVOLUTION_VOLUME::initialize()
{
  return initialize(Parameters());
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
bool CONVOLUTION_VOLUME::initialize(const Parameters& config)
{
  // Abort if layer is already initialized
  if (Base::setupRequired())
  {
    if (Base::isInitialized())
    {
      FFNN_WARN_NAMED("layer::ConvolutionVolume", "<" << Base::getID() << "> already initialized.");
      return false;
    }
    else
    {
      /// Reset all filters
      reset(config);
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
void CONVOLUTION_VOLUME::reset(const Parameters& config)
{
  // Deduce filter height
  const SizeType f_h =
    EmbeddingMode == ColEmbedding ?
    (Base::input_shape_.height * Base::input_shape_.depth) : Base::input_shape_.height;

  // Deduce filter width
  const SizeType f_w =
    EmbeddingMode == RowEmbedding ?
    (Base::input_shape_.width * Base::input_shape_.depth) : Base::input_shape_.width;

  // Initializer all filters
  bank_.setRandom(f_h, f_w);
  bank_ *= config.init_weight_std;
  if (std::abs(config.init_weight_mean) > 0)
  {
    bank_ += config.init_weight_mean;
  }

  // Set uniformly random bias matrix + add biases
  b_.setRandom(Base::output_shape_.depth, 1);
  b_ *= config.init_bias_std;
  if (std::abs(config.init_bias_mean) > 0)
  {
    b_.array() += config.init_bias_mean;
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
template<typename InputBlockType, typename OutputBlockType>
void CONVOLUTION_VOLUME::forward(const Eigen::MatrixBase<InputBlockType>& input,
                                 Eigen::MatrixBase<OutputBlockType> const& output)
{
  // @see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
  auto _output = const_cast<Eigen::MatrixBase<OutputBlockType>&>(output);

  // Multiply all filters
  BiasVectorType filtered(Base::output_shape_.depth, 1);
  for (size_t idx = 0; idx < bank_.filters.size(); idx++)
  {
    filtered(idx) = (input.array() * bank_.filters[idx].array()).sum();
  }

  // @see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
  _output = filtered;
  _output += b_;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
template<typename InputBlockType, typename ForwardErrorBlockType>
void CONVOLUTION_VOLUME::backward(const Eigen::MatrixBase<InputBlockType>& input,
                                  const Eigen::MatrixBase<ForwardErrorBlockType>& error)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void CONVOLUTION_VOLUME::save(typename CONVOLUTION_VOLUME::OutputArchive& ar,
                              typename CONVOLUTION_VOLUME::VersionType version) const
{
  ffnn::io::signature::apply<CONVOLUTION_VOLUME>(ar);
  Base::save(ar, version);

  // Save filters

  // Save bias matrix
  ar & b_;

  FFNN_DEBUG_NAMED("layer::ConvolutionVolume", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         FFNN_SIZE_TYPE EmbeddingMode>
void CONVOLUTION_VOLUME::load(typename CONVOLUTION_VOLUME::InputArchive& ar,
                            typename CONVOLUTION_VOLUME::VersionType version)
{
  ffnn::io::signature::check<CONVOLUTION_VOLUME>(ar);
  Base::load(ar, version);

  // Load filters

  // Load bias matrix
  ar & b_;

  FFNN_DEBUG_NAMED("layer::ConvolutionVolume", "Loaded");
}

#undef CONVOLUTION_VOLUME_TARGS
#undef CONVOLUTION_VOLUME
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP
