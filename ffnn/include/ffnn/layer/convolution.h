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

// FFNN Optimization
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace layer
{
#define TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterCountAtCompileTime,\
  StrideAtCompileTime,\
  Mode,\
  _HiddenLayerShape

#define VOLUME_TARGS\
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
          typename _HiddenLayerShape =
            hidden_layer_shape<
              embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime),
              embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime),
              embed_dimension<Mode, ColEmbedding>(output_dimension(HeightAtCompileTime, FilterHeightAtCompileTime, StrideAtCompileTime), FilterCountAtCompileTime),
              embed_dimension<Mode, RowEmbedding>(output_dimension(WidthAtCompileTime,  FilterWidthAtCompileTime,  StrideAtCompileTime), FilterCountAtCompileTime)>>
class Convolution :
  public Hidden<ValueType, _HiddenLayerShape>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, _HiddenLayerShape>;

  /// Self type alias
  using Self = Convolution<TARGS>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Receptive-volume type standardization
  typedef ConvolutionVolume<VOLUME_TARGS> ParametersType;

  /// Forward mapping bank standardization
  typedef boost::multi_array<ValueType*, 2> ForwardMapType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

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
   * @retval true  if layer was initialized successfully
   * @retval false otherwise
   *
   * @warning If layer is not loaded instance, this method will initialize layer sizings
   *          but weights and biases will be zero
   */
  bool initialize();

  /**
   * @brief Initialize layer weights and biases according to particular distributions
   * @param wd  distribution to sample for connection weights
   * @param bd  distribution to sample for biases
   * @retval true  if layer was initialized successfully
   * @retval false otherwise
   *
   * @warning If layer is a loaded instance, this method will initialize layer sizings
   *          but weights will not be reset according to the given distributions
   */
  template<typename WeightDistribution, typename BiasDistribution>
  bool initialize(const WeightDistribution& wd, const BiasDistribution& bd);

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
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   * @warning <code>backward</code> and <code>update</code> methods are expected to throw if an
   *          optimizer has not been set explicitly
   */
  void setOptimizer(typename Optimizer::Ptr opt);

  inline const ParametersType& getParameters() const
  {
    return parameters_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(Convolution)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  //FFNN_REGISTER_OPTIMIZER(Convolution, Adam);
  FFNN_REGISTER_OPTIMIZER(Convolution, GradientDescent);

  /**
   * @brief Reset all internal volumes
   */
  void reset();

  /// Layer configuration parameters
  ParametersType parameters_;

  ForwardMapType forward_error_mappings_;

  ForwardMapType output_mappings_;

  /// "True" shape of the ouput with no depth-embedding
  ShapeType input_volume_shape_;

  /// "True" shape of the ouput with no depth-embedding
  ShapeType output_volume_shape_;

  /// Shape of receptive fields
  ShapeType filter_shape_;

  /// Stride between receptive fields
  ShapeType filter_stride_;

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
#undef TARGS
#undef VOLUME_TARGS
#endif  // FFNN_LAYER_CONVOLUTION_H
