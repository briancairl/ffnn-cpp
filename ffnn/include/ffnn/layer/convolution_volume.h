/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_VOLUME_H
#define FFNN_LAYER_CONVOLUTION_VOLUME_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN (internal)
#include <ffnn/layer/internal/shape.h>
#include <ffnn/layer/internal/interface.h>

namespace ffnn
{
namespace layer
{
typedef enum
{
  RowEmbedding = 0, ///< Embed depth along filter matrix rows
  ColEmbedding = 1, ///< Embed depth along filter matrix cols
}
EmbeddingMode;

template<EmbeddingMode mode, EmbeddingMode ref, typename SizeType>
constexpr SizeType embed_dimension(SizeType n, SizeType m)
{
  return (mode == ref) ? internal::multiply_if_not_dynamic_sizes<SizeType>(n, m) : n;
}

template<typename SizeType>
constexpr SizeType output_dimension(SizeType n, SizeType fn, SizeType stride)
{
  return (n - fn) / stride + 1;
}


template<typename KernelMatrixType>
class FilterBank :
  public std::conditional
  <
    internal::is_alignable_128<KernelMatrixType>::value,
    std::vector<KernelMatrixType, Eigen::aligned_allocator<typename KernelMatrixType::Scalar>>,
    std::vector<KernelMatrixType>
  >::type
{
public:
  typedef typename KernelMatrixType::Scalar ScalarType;

  typedef typename KernelMatrixType::Index SizeType;

  typedef typename KernelMatrixType::Index OffsetType;

  explicit
  FilterBank(SizeType filter_count)
  {
    this->resize(filter_count);
  }

  void setZero(SizeType height, SizeType width)
  {
    for (auto& filter : *this)
    {
      filter.setZero(height, width);
    }
  }

  FilterBank& operator*=(ScalarType scalar)
  {
    for (auto& filter : *this)
    {
      filter.array() *= scalar;
    }
    return *this;
  }

  FilterBank& operator+=(ScalarType scalar)
  {
    for (auto& filter : *this)
    {
      filter.array() += scalar;
    }
    return *this;
  }
};

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
  typedef Eigen::Matrix<ValueType,
                        embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime),
                        embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime),
                        Mode == ColEmbedding ? Eigen::ColMajor : Eigen::RowMajor> KernelMatrixType;

  /// Bias vector type standardization
  typedef Eigen::Matrix<ValueType, FilterCountAtCompileTime, 1, Eigen::ColMajor> BiasVectorType;

  /// Filter collection type standardization
  typedef FilterBank<KernelMatrixType> FilterBankType;

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
   * @brief Performs backward error propagation
   * @param input  a block (matrix; depth embedded) of input values
   * @param[out] backward_error  a block (matrix; depth embedded) of previous layer output error values
   */
  template<typename InputBlockType, typename BackwardErrorBlockPtr>
  void backward(const Eigen::Block<InputBlockType>& input, BackwardErrorBlockPtr backward_error_ptr);

  /**
   * @brief Exposes internal biasing weights
   * @return input-biasing vector
   */
  inline const BiasVectorType& getBiases() const
  {
    return b_;
  }

  /**
   * @brief Exposes internal filters
   * @return filter collection
   */
  inline const FilterBankType& getFilters() const
  {
    return filters_;
  }

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

private:
  /**
   * @brief Reset filter weights and biases
   */
  void reset();

  /**
   * @brief Unused; does nothing
   */
  bool initialize();

  /// Filter bank for receptive field
  FilterBankType filters_;

  /// Bias vector
  BiasVectorType b_;

  // Output mapping
  ValueType* output_ptr_;

  // Backward-error mapping
  ValueType* forward_error_ptr_;

  /// Number of filters associated with the field
  SizeType filter_count_;

public:
  FFNN_REGISTER_SERIALIZABLE(ConvolutionVolume)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(internal::is_alignable_128<BiasVectorType>::value);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/convolution_volume.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_VOLUME_H
