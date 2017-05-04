/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_RECEPTIVE_VOLUME_H
#define FFNN_LAYER_RECEPTIVE_VOLUME_H

// C++ Standard Library
#include <vector>

// FFNN (internal)
#include <ffnn/layer/internal/dimensions.h>
#include <ffnn/layer/internal/interface.h>

namespace ffnn
{
namespace layer
{
#define IS_DYNAMIC_PAIR(n, m) (IS_DYNAMIC(n) || IS_DYNAMIC(m))
#define PROD_IF_STATIC_PAIR(n, m) (IS_DYNAMIC_PAIR(n, m) ? Eigen::Dynamic : (n*m))

enum EmbeddingMode
{
  RowEmbedding = 0,
  ColEmbedding = 1,
};

#define RECEPTIVE_VOLUME_TARGS ValueType,\
                               HeightAtCompileTime,\
                               WidthAtCompileTime,\
                               DepthAtCompileTime,\
                               FilterCountAtCompileTime,\
                               EmbeddingMode

template <typename ValueType,
          FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterCountAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE EmbeddingMode = ColEmbedding>
class ReceptiveVolume :
  public internal::Interface<ValueType>
{
public:
  /// Base type alias
  using Base = internal::Interface<ValueType>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size-type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::DimType DimType;

  /// Filter kernel matrix standardization
  typedef Eigen::Matrix<ValueType,
                        EmbeddingMode == ColEmbedding ? PROD_IF_STATIC_PAIR(HeightAtCompileTime, DepthAtCompileTime) : HeightAtCompileTime,
                        EmbeddingMode == RowEmbedding ? PROD_IF_STATIC_PAIR(WidthAtCompileTime,  DepthAtCompileTime) : WidthAtCompileTime,
                        EmbeddingMode == ColEmbedding ? Eigen::ColMajor : Eigen::RowMajor> KernelMatrixType;

  /// Bias vector type standardization
  typedef Eigen::Matrix<ValueType, FilterCountAtCompileTime, 1, Eigen::ColMajor> BiasVectorType;

  /// Filter collection type standardization
  typedef std::vector<KernelMatrixType, Eigen::aligned_allocator<ValueType>> FilterBankType;

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
  ReceptiveVolume(const DimType& input_dim = DimType(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
                  SizeType filter_count = FilterCountAtCompileTime,
                  const Parameters& config = Parameters());
  virtual ~ReceptiveVolume();

  /**
   * @brief Initialize the volume
   */
  bool initialize();

  /**
   * @brief Reset filter weights and biases
   */
  void reset();

  template<typename InputBlockType, typename OutputBlockType>
  void forward(const Eigen::MatrixBase<InputBlockType>& input,
               Eigen::MatrixBase<OutputBlockType> const& output);

  template<typename InputBlockType, typename ForwardErrorBlockType>
  void backward(const Eigen::MatrixBase<InputBlockType>& input,
                const Eigen::MatrixBase<ForwardErrorBlockType>& error);

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
    return filter_bank_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(ReceptiveVolume)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  /// Configuration struct
  Parameters config_;

  /// Filter bank for receptive field
  FilterBankType filter_bank_;

  /// Bias vector
  BiasVectorType b_;

  /// Number of filters associated with the field
  SizeType filter_count_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/receptive_volume.hpp>
#endif  // FFNN_LAYER_RECEPTIVE_VOLUME_H
