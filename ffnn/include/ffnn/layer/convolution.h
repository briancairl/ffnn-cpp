/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// Boost
#include "boost/multi_array.hpp"

// FFNN Layer
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/receptive_volume.h>
#include <ffnn/neuron/neuron.h>

// FFNN Optimization
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace layer
{
#define CONVOLUTION_TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterCountAtCompileTime,\
  StrideAtCompileTime,\
  EmbeddingMode

#define CONVOLUTION_VOLUME_TARGS\
  ValueType,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterCountAtCompileTime,\
  EmbeddingMode

#define RESOLVE_CONVOLUTION_OUTPUT(n, fn, s) ((n - fn) / s + 1)
#define CONVOLUTION_OUTPUT_HEIGHT RESOLVE_CONVOLUTION_OUTPUT(HeightAtCompileTime, FilterHeightAtCompileTime, StrideAtCompileTime)
#define CONVOLUTION_OUTPUT_WIDTH  RESOLVE_CONVOLUTION_OUTPUT(WidthAtCompileTime, FilterWidthAtCompileTime, StrideAtCompileTime)

#define CONVOLUTION_BASE_TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  CONVOLUTION_OUTPUT_HEIGHT,\
  CONVOLUTION_OUTPUT_WIDTH

/**
 * @brief A convolution layer
 */
template <typename ValueType,
          FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterHeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterWidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterCountAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE StrideAtCompileTime = 1,
          FFNN_SIZE_TYPE EmbeddingMode = ColEmbedding>
class Convolution :
  public Hidden<CONVOLUTION_BASE_TARGS>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType,
                      HeightAtCompileTime,
                      WidthAtCompileTime,
                      CONVOLUTION_OUTPUT_HEIGHT,
                      CONVOLUTION_OUTPUT_WIDTH>;

  /// Self type alias
  using Self = Convolution<CONVOLUTION_TARGS>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Receptive-volume type standardization
  typedef ReceptiveVolume<CONVOLUTION_VOLUME_TARGS> ReceptiveVolumeType;

  /// Recptive-volume bank standardization
  typedef boost::multi_array<typename ReceptiveVolumeType::Ptr, 2> ReceptiveVolumeBankType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// Configuration struct type alias
  typedef typename ReceptiveVolumeType::Parameters Parameters;

  /**
   * @brief Setup constructor
   * @param output_size  number of layer outputs
   * @param config  layer configuration struct
   */
  explicit
  Convolution(const ShapeType& input_shape = ShapeType(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
              const SizeType& filter_height = FilterHeightAtCompileTime,
              const SizeType& filter_width = FilterWidthAtCompileTime,
              const SizeType& filter_count = FilterCountAtCompileTime,
              const SizeType& filter_stride = StrideAtCompileTime,
              const Parameters& config = Parameters());
  virtual ~Convolution();

  /**
   * @brief Initialize the layer
   */
  bool initialize();

  /**
   * @brief Performs forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   * @warning Does not apply layer weight updates
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  bool backward();

  /**
   * @brief Applies accumulated layer weight updates computed during optimization
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  bool update();

  /**
   * @brief Reset all internal volumes
   */
  void reset();

  /**
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   * @warning <code>backward</code> and <code>update</code> methods are expected to throw if an
   *          optimizer has not been set explicitly
   */
  void setOptimizer(typename Optimizer::Ptr opt);

  inline const ReceptiveVolumeBankType& getReceptiveVolumes() const
  {
    return receptors_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(Convolution)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  //FFNN_REGISTER_OPTIMIZER(Convolution, Adam);
  //FFNN_REGISTER_OPTIMIZER(Convolution, GradientDescent);

  /// Parameters config
  Parameters config_;

  /// Layer configuration parameters
  ReceptiveVolumeBankType receptors_;

  /// Shape of receptive fields
  ShapeType filter_shape_;

  /// Shape number of filters per recpetive field
  SizeType filter_count_;

  /// Stride between receptive fields
  SizeType filter_stride_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> type by default
   * @see   setOptimizer
   */
  typename Optimizer::Ptr opt_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/convolution.hpp>

#undef CONVOLUTION_OUTPUT_HEIGHT
#undef CONVOLUTION_OUTPUT_WIDTH
#undef CONVOLUTION_BASE_TARGS
#undef CONVOLUTION_VOLUME_TARGS
#undef CONVOLUTION_TARGS
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
