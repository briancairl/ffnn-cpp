/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_VOLUME_H
#define FFNN_LAYER_CONVOLUTION_VOLUME_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/layer/convolution_defs.h>

// FFNN (internal)
#include <ffnn/layer/internal/shape.h>
#include <ffnn/layer/internal/interface.h>
#include <ffnn/layer/internal/filter_bank.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE FilterCountAtCompileTime = Eigen::Dynamic,
         EmbeddingMode Mode = ColEmbedding>
class ConvolutionVolume :
  public internal::Interface<ValueType>
{
public:
  /// Base type alias
  using Base = internal::Interface<ValueType>;

  /// Self type alias
  using Self = ConvolutionVolume<ValueType,
                                 HeightAtCompileTime,
                                 WidthAtCompileTime,
                                 DepthAtCompileTime,
                                 FilterCountAtCompileTime,
                                 Mode>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size-type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Filter kernel matrix standardization
  typedef internal::FilterBank<ValueType,
                                embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime),
                                embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime),
                                Mode == ColEmbedding ? Eigen::ColMajor : Eigen::RowMajor> FilterBankType;

  /**
   * @brief
   */
  ConvolutionVolume(const ShapeType& filter_shape = ShapeType(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
                    const SizeType& filter_count  = FilterCountAtCompileTime);
  virtual ~ConvolutionVolume();

  /**
   * @brief Initialize volume weights and biases according to particular distributions
   * @param wd  distribution to sample for connection weights
   * @param bd  distribution to sample for biases
   * @retval false otherwise
   *
   * @warning If volume is a loaded instance, this method will initialize volume sizings
   *          but weights will not be reset according to the given distributions
   */
  template<typename WeightDistribution, typename BiasDistribution>
  bool initialize(const WeightDistribution& wd, const BiasDistribution& bd);

  /**
   * @brief Performs forward value propagation
   * @param input  a block (matrix; depth embedded) of input values
   */
  template<typename InputBlockType>
  void forward(const Eigen::Block<InputBlockType>& input);

  /**
   * @brief Sets memory map to contiguous output buffer
   */
  inline void setOutputMapping(ValueType* const ptr)
  {
    output_ptr_ = ptr;
  }

  /**
   * @brief Sets memory map to contiguous backward-error buffer
   */
  inline void setForwardErrorMapping(ValueType* const ptr)
  {
    forward_error_ptr_ = ptr;
  }

  /**
   * @brief Gets memory map to contiguous backward-error buffer
   */
  inline const ValueType* getForwardErrorMapping() const
  {
    return forward_error_ptr_;
  }

  /// Filter bank for receptive field
  FilterBankType filters;

private:
  /**
   * @brief Reset filter weights and biases
   */
  void reset();

  /**
   * @brief Unused; does nothing
   */
  bool initialize();

  // Output mapping
  ValueType* output_ptr_;

  // Backward-error mapping
  ValueType* forward_error_ptr_;

public:
  FFNN_REGISTER_SERIALIZABLE(ConvolutionVolume)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/convolution_volume.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_VOLUME_H
