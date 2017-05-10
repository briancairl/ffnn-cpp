/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_H
#define FFNN_LAYER_CONVOLUTION_H

// Boost
#include "boost/multi_array.hpp"

// FFNN Layer
#include <ffnn/layer/layer.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/convolution_volume.h>
#include <ffnn/neuron/neuron.h>

// FFNN Optimization
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace layer
{
#define CONV_TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterCountAtCompileTime,\
  StrideAtCompileTime,\
  Mode,\
  _HSize

#define CONV_VOLUME_TARGS\
  ValueType,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterCountAtCompileTime,\
  Mode

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
          EmbeddingMode Mode = ColEmbedding,
          class _HSize =
            hidden_size_evaluator<
              embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime),
              embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime),
              embed_dimension<Mode, ColEmbedding>(output_dimension(HeightAtCompileTime, FilterHeightAtCompileTime, StrideAtCompileTime), FilterCountAtCompileTime),
              embed_dimension<Mode, RowEmbedding>(output_dimension(WidthAtCompileTime,  FilterWidthAtCompileTime,  StrideAtCompileTime), FilterCountAtCompileTime)>>
class Convolution :
  public Hidden<ValueType, _HSize::input_height, _HSize::input_width, _HSize::output_height, _HSize::output_width>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, _HSize::input_height, _HSize::input_width, _HSize::output_height, _HSize::output_width>;

  /// Self type alias
  using Self = Convolution<CONV_TARGS>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Receptive-volume type standardization
  typedef ConvolutionVolume<CONV_VOLUME_TARGS> ConvolutionVolumeType;

  /// Recptive-volume bank standardization
  typedef boost::multi_array<ConvolutionVolumeType, 2> ConvolutionVolumeBankType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// Configuration struct type alias
  typedef typename ConvolutionVolumeType::Parameters Parameters;

  /**
   * @brief Setup constructor
   */
  explicit
  Convolution(const ShapeType& input_shape = ShapeType(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
              const SizeType& filter_height = FilterHeightAtCompileTime,
              const SizeType& filter_width = FilterWidthAtCompileTime,
              const SizeType& filter_count = FilterCountAtCompileTime,
              const SizeType& filter_stride = StrideAtCompileTime);
  virtual ~Convolution();

  /**
   * @brief Initialize the layer
   */
  bool initialize();
  bool initialize(const Parameters& config);

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
  void reset(const Parameters& config = Parameters());

  /**
   * @brief Computes previous layer error from current layer output error
   */ 
  bool computeBackwardError();

  /**
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   * @warning <code>backward</code> and <code>update</code> methods are expected to throw if an
   *          optimizer has not been set explicitly
   */
  void setOptimizer(typename Optimizer::Ptr opt);

  inline const ConvolutionVolumeBankType& getConvolutionVolumes() const
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

  /// Layer configuration parameters
  ConvolutionVolumeBankType receptors_;

  /// "True" shape of the ouput with no depth-embedding
  ShapeType input_volume_shape_;

  /// "True" shape of the ouput with no depth-embedding
  ShapeType output_volume_shape_;

  /// Shape of receptive fields
  ShapeType filter_shape_;

  /// Stride between receptive fields
  SizeType filter_stride_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> type by default
   * @see   setOptimizer
   */
  typename Optimizer::Ptr opt_;

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_shape_.size()</code>
   */
  OffsetType connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/convolution.hpp>

// Cleanup definitions
#undef CONV_TARGS
#undef CONV_VOLUME_TARGS
#endif  // FFNN_LAYER_CONVOLUTION_H
