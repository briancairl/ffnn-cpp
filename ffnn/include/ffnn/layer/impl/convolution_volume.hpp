/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP
#define FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP

// C++ Standard Library
#include <exception>

// Boost
#include <boost/bind.hpp>

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
                                 Mode
#define CONVOLUTION_VOLUME ConvolutionVolume<CONVOLUTION_VOLUME_TARGS>

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
CONVOLUTION_VOLUME::ConvolutionVolume(const ShapeType& filter_shape, const SizeType& filter_count) :
  Base(filter_shape, ShapeType(1, 1, filter_count)),
  filters_(filter_count)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
CONVOLUTION_VOLUME::~ConvolutionVolume()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::ConvolutionVolume", "Destroying [layer::ConvolutionVolume] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
bool CONVOLUTION_VOLUME::initialize()
{
  throw std::logic_error("ConvolutionVolume objects should not be initialized this way.");
  return false;
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
template<typename WeightDistribution,
         typename BiasDistribution>
bool CONVOLUTION_VOLUME::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
{
  // Abort if layer is already initialized
  if (Base::setupRequired())
  {
    if (Base::isInitialized())
    {
      FFNN_WARN_NAMED("layer::ConvolutionVolume", "<" << Base::getID() << "> already initialized.");
      return false;
    }

    /// Reset all filters
    reset();

    // Set filter connections weights
    {
      auto coeffInitfn = [](ValueType x, const WeightDistribution& dist)
      {
        return dist.generate();
      };
      for (auto& filter : filters_)
      {
        filter = filter.unaryExpr(boost::bind<ValueType>(coeffInitfn, _1, wd));
      }
    }

    // Set layer biases
    {
      auto coeffInitfn = [](ValueType x, const BiasDistribution& dist)
      {
        return dist.generate();
      };
      b_ = b_.unaryExpr(boost::bind<ValueType>(coeffInitfn, _1, bd));
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
         EmbeddingMode Mode>
void CONVOLUTION_VOLUME::reset()
{
  // Initiliaze all filters and biases
  filters_.setZero(embed_dimension<Mode, ColEmbedding>(Base::input_shape_.height, Base::input_shape_.depth),
                   embed_dimension<Mode, RowEmbedding>(Base::input_shape_.width,  Base::input_shape_.depth));
  b_.setZero(Base::output_shape_.depth, 1);
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
template<typename InputBlockType>
void CONVOLUTION_VOLUME::forward(const Eigen::MatrixBase<InputBlockType>& input)
{
  // Multiply all filters
  for (OffsetType idx = 0; idx < Base::output_shape_.depth; idx++)
  {
    output_ptr_[idx]  = input.cwiseProduct(filters_[idx]).sum() + b_(idx);
    output_ptr_[idx] += b_(idx);
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
template<typename InputBlockType, typename ForwardErrorBlockType>
void CONVOLUTION_VOLUME::backward(const Eigen::MatrixBase<InputBlockType>& input,
                                  const Eigen::MatrixBase<ForwardErrorBlockType>& error)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
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
         EmbeddingMode Mode>
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
