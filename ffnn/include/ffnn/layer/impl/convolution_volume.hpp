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
#define TARGS ValueType, HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime, FilterCountAtCompileTime, Mode

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
ConvolutionVolume<TARGS>::ConvolutionVolume(const ShapeType& filter_shape, const SizeType& filter_count) :
  Base(filter_shape, ShapeType(1, 1, filter_count))
{}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
ConvolutionVolume<TARGS>::~ConvolutionVolume()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::ConvolutionVolume", "Destroying [layer::ConvolutionVolume] object <" << this->getID() << ">");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
bool ConvolutionVolume<TARGS>::initialize()
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
bool ConvolutionVolume<TARGS>::initialize(const WeightDistribution& wd, const BiasDistribution& bd)
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
    auto fn = [](ValueType x, const WeightDistribution& dist) {return dist.generate();};
    for (auto& filter : filters)
    {
      filter.kernel = filter.kernel.unaryExpr(boost::bind<ValueType>(fn, _1, wd));
      filter.bias = bd.generate();
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
void ConvolutionVolume<TARGS>::reset()
{
  // Resize bank
  filters.resize(Base::output_shape_.depth);

  // Initiliaze all filters and biases
  filters.setZero(embed_dimension<Mode, ColEmbedding>(Base::input_shape_.height, Base::input_shape_.depth),
                  embed_dimension<Mode, RowEmbedding>(Base::input_shape_.width,  Base::input_shape_.depth));
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
template<typename InputBlockType>
void ConvolutionVolume<TARGS>::forward(const Eigen::Block<InputBlockType>& input)
{
  // Multiply all filters
  for (OffsetType idx = 0; idx < Base::output_shape_.depth; idx++)
  {
    output_ptr_[idx]  = input.cwiseProduct(filters[idx].kernel).sum();
    output_ptr_[idx] += filters[idx].bias;
  }
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
void ConvolutionVolume<TARGS>::save(typename ConvolutionVolume<TARGS>::OutputArchive& ar,
                                    typename ConvolutionVolume<TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<ConvolutionVolume<TARGS>>(ar);
  Base::save(ar, version);

  // Save filters
  ar & filters;

  FFNN_DEBUG_NAMED("layer::ConvolutionVolume", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime,
         FFNN_SIZE_TYPE WidthAtCompileTime,
         FFNN_SIZE_TYPE DepthAtCompileTime,
         FFNN_SIZE_TYPE FilterCountAtCompileTime,
         EmbeddingMode Mode>
void ConvolutionVolume<TARGS>::load(typename ConvolutionVolume<TARGS>::InputArchive& ar,
                                    typename ConvolutionVolume<TARGS>::VersionType version)
{
  ffnn::io::signature::check<ConvolutionVolume<TARGS>>(ar);
  Base::load(ar, version);

  // Load filters
  ar & filters;

  FFNN_DEBUG_NAMED("layer::ConvolutionVolume", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_CONVOLUTION_VOLUME_HPP
