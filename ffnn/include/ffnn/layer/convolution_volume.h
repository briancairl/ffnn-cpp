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
template<FFNN_SIZE_TYPE SizeN>
struct is_dynamic :
  std::conditional<SizeN == Eigen::Dynamic, std::true_type, std::false_type>::type
{};


#define IS_DYNAMIC_PAIR(n, m) (IS_DYNAMIC(n) || IS_DYNAMIC(m))
#define PROD_IF_STATIC_PAIR(n, m) (IS_DYNAMIC_PAIR(n, m) ? Eigen::Dynamic : (n*m))
#define CONV_EMBEDDED_H(h, d) EmbeddingMode == ColEmbedding ? PROD_IF_STATIC_PAIR(h, d) : h
#define CONV_EMBEDDED_W(w, d) EmbeddingMode == RowEmbedding ? PROD_IF_STATIC_PAIR(w, d) : w



enum EmbeddingMode
{
  RowEmbedding = 0, ///< Embed depth along filter matrix rows
  ColEmbedding = 1, ///< Embed depth along filter matrix cols
};

template<class BlockType>
struct is_alignable_128 :
  std::conditional<(sizeof(BlockType)%16) == 0, std::true_type, std::false_type>::type
{};


template<typename KernelMatrixType>
class FilterBank :
  public std::conditional
  <
    is_alignable_128<KernelMatrixType>::value,
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

  void setRandom(SizeType height, SizeType width)
  {
    for (auto& filter : *this)
    {
      filter.setRandom(height, width);
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
         FFNN_SIZE_TYPE EmbeddingMode = ColEmbedding>
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
                                 EmbeddingMode>;

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
                        CONV_EMBEDDED_H(HeightAtCompileTime, DepthAtCompileTime),
                        CONV_EMBEDDED_W(WidthAtCompileTime, DepthAtCompileTime),
                        EmbeddingMode == ColEmbedding ? Eigen::ColMajor : Eigen::RowMajor> KernelMatrixType;

  /// Bias vector type standardization
  typedef Eigen::Matrix<ValueType, FilterCountAtCompileTime, 1, Eigen::ColMajor> BiasVectorType;

  /// Filter collection type standardization
  typedef FilterBank<KernelMatrixType> FilterBankType;

  /// A configuration object for a FullyConnected hidden layer
  struct Parameters
  {
    /// Standard deviation of connection weights on init
    ScalarType init_weight_std;

    /// Standard deviation of biases on init
    ScalarType init_bias_std;

    /// Connection weight mean (bias) on init
    ScalarType init_weight_mean;

    /// Connection biasing mean (bias) on init
    ScalarType init_bias_mean;

    /**
     * @brief Setup constructor
     * @param init_weight_std  Standard deviation of initial weights
     * @param init_bias_std  Standard deviation of initial weights
     * @param init_weight_mean  Mean of intial weights
     * @param init_bias_mean  Mean of intial biases
     */
    explicit
    Parameters(ScalarType init_weight_std = 1e-3,
               ScalarType init_bias_std = 1e-3,
               ScalarType init_weight_mean = 0.0,
               ScalarType init_bias_mean = 0.0);
  };

  /**
   * @brief
   */
  ConvolutionVolume(const ShapeType& filter_shape = ShapeType(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
                    const SizeType& filter_count = FilterCountAtCompileTime);
  virtual ~ConvolutionVolume();

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
  inline void setBackwardErrorMapping(ValueType* const ptr)
  {
    backward_error_ptr_ = ptr;
  }

  /**
   * @brief Initialize the volume
   */
  bool initialize();
  bool initialize(const Parameters& config);

  /**
   * @brief Reset filter weights and biases
   * @param config  parameter reset configuration
   */
  void reset(const Parameters& config = Parameters());

  /**
   * @brief Performs forward value propagation
   * @param input  a block (matrix; depth embedded) of input values
   */
  template<typename InputBlockType>
  void forward(const Eigen::MatrixBase<InputBlockType>& input);

  /**
   * @brief Performs backward error propagation
   * @param input  a block (matrix; depth embedded) of input values
   * @param forward_error  a block (matrix; depth embedded) of layer-output error values
   */
  template<typename InputBlockType, typename ForwardErrorBlockType>
  void backward(const Eigen::MatrixBase<InputBlockType>& input,
                const Eigen::MatrixBase<ForwardErrorBlockType>& forward_error);

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

  FFNN_REGISTER_SERIALIZABLE(ConvolutionVolume)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  /// Configuration struct
  Parameters config_;

  /// Filter bank for receptive field
  FilterBankType filters_;

  /// Bias vector
  BiasVectorType b_;

  // Output mapping
  ValueType* output_ptr_;

  // Backward-error mapping
  ValueType* backward_error_ptr_;

  /// Number of filters associated with the field
  SizeType filter_count_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(is_alignable_128<BiasVectorType>::value);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/convolution_volume.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_VOLUME_H
