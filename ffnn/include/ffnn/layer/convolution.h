/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_H
#define FFNN_LAYER_CONVOLUTION_H

// Boost
#include "boost/multi_array.hpp"

// FFNN Layer
#include <ffnn/distribution/distribution.h>
#include <ffnn/distribution/normal.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/layer/convolution/defs.h>
#include <ffnn/layer/convolution/filter.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A convolution layer
 */
template <typename ValueType,
          ffnn::size_type HeightAtCompileTime = Eigen::Dynamic,
          ffnn::size_type WidthAtCompileTime = Eigen::Dynamic,
          ffnn::size_type DepthAtCompileTime = Eigen::Dynamic,
          ffnn::size_type FilterHeightAtCompileTime = Eigen::Dynamic,
          ffnn::size_type FilterWidthAtCompileTime = Eigen::Dynamic,
          ffnn::size_type FilterDepthAtCompileTime = Eigen::Dynamic,
          ffnn::size_type StrideAtCompileTime = 1,
          EmbeddingMode Mode = ColEmbedding,
          typename _HiddenLayerShape =
            hidden_layer_shape<
              convolution::embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime),
              convolution::embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime),
              convolution::embed_dimension<Mode, ColEmbedding>(convolution::output_dimension(HeightAtCompileTime, FilterHeightAtCompileTime, StrideAtCompileTime), FilterDepthAtCompileTime),
              convolution::embed_dimension<Mode, RowEmbedding>(convolution::output_dimension(WidthAtCompileTime,  FilterWidthAtCompileTime,  StrideAtCompileTime), FilterDepthAtCompileTime)>>
class Convolution :
  public Hidden<ValueType, _HiddenLayerShape>
{
public:
  /// Base type alias
  using BaseType = Hidden<ValueType, _HiddenLayerShape>;

  /// Self type alias
  using SelfType = Convolution<ValueType,
                               HeightAtCompileTime,
                               WidthAtCompileTime,
                               DepthAtCompileTime,
                               FilterHeightAtCompileTime,
                               FilterWidthAtCompileTime,
                               FilterDepthAtCompileTime,
                               StrideAtCompileTime,
                               Mode,
                               _HiddenLayerShape>;

  /// Scalar type standardization
  typedef typename BaseType::ScalarType ScalarType;

  /// Size type standardization
  typedef typename BaseType::SizeType SizeType;

  /// Offset type standardization
  typedef typename BaseType::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Receptive-volume type standardization
  typedef Filter<ValueType,
                 convolution::embed_dimension<Mode, ColEmbedding>(FilterHeightAtCompileTime, DepthAtCompileTime),
                 convolution::embed_dimension<Mode, RowEmbedding>(FilterWidthAtCompileTime,  DepthAtCompileTime),
                 (Mode == ColEmbedding) ? Eigen::ColMajor : Eigen::RowMajor>
                 FilterType;

  /// 2D-value mapping standardization
  typedef boost::multi_array<ValueType*, 2> ForwardMapType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<SelfType> Optimizer;

  /// Dsitribution standardization
  typedef distribution::Distribution<ScalarType> Distribution;

  /// Layer configuration struct
  class Configuration
  {
  public:
    friend SelfType;

    Configuration() :
      input_shape_(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime),
      filter_shape_(FilterHeightAtCompileTime, FilterWidthAtCompileTime, FilterDepthAtCompileTime),
      parameter_dist_(boost::make_shared<typename distribution::StandardNormal<SelfType>>())
      opt_(boost::make_shared<typename optimizer::None<SelfType>>())
    {
      /// Set defaults shape options
      setShapeOptions(input_shape_, filter_shape_, 1)
    }

    inline Configuration& setOptimizer(const typename Optimizer::Ptr& opt)
    {
      opt_ = opt;
      return *this;
    }

    inline Configuration& setParameterDistribution(const typename Distribution::Ptr& opt)
    {
      opt_ = opt;
      return *this;
    }

    inline Configuration& setShapeOptions(const ShapeType& input_shape,
                                          const ShapeType& filter_shape,
                                          const SizeType stride)
    {
      // Set layer input shape
      input_shape_ = input_shape;

      // Setup output shape before depth embdedding
      const SizeType h_proto = convolution::output_dimension(input_shape.height, filter_shape.height, stride);
      const SizeType w_proto = convolution::output_dimension(input_shape.width, filter_shape.width, stride);

      // Set depth-embedded output shape
      output_shape_.width  = convolution::embed_dimension<Mode, ColEmbedding>(h_proto, filter_shape.depth);
      output_shape_.height = convolution::embed_dimension<Mode, RowEmbedding>(w_proto, filter_shape.depth);
      output_shape_.depth  = 1;
      return *this;
    }

    inline Configuration& setFilterOptions(const ShapeType& shape, SizeType stride = 1)
    {
      return *this;
    }

  private:
    /// Shape of layer input
    ShapeType input_shape_;

    /// Shape of layer output
    ShapeType output_shape_;

    /// Shape of receptive fields
    ShapeType filter_shape_;

    /// Stride between receptive fields
    ShapeType stride_shape_;

    /// Embdedded shape of layer input
    ShapeType embedded_input_shape_;

    /// Embdedded shape of layer output
    ShapeType embedded_output_shape_;

    /**
     * @brief Distribution of initializing layer parameters
     */
    typename Distribution::Ptr parameter_dist_;

    /**
     * @brief Weight optimization resource
     * @note  This will be the <code>optimizer::None</code> type by default
     * @see   setOptimizer
     */
    typename Optimizer::Ptr opt_;
  };

  /**
   * @brief Setup constructor
   */
  explicit Convolution(const Configuration& config = Configuration());
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

  /// User layer configurations
  Configuration config_;

  /// Shape of the ouput with no depth-embedding
  ShapeType input_volume_shape_;

  /// Shape of the ouput with no depth-embedding
  ShapeType output_volume_shape_;

  /// Layer configuration parameters
  ParametersType parameters_;

  /// Forward error mapping grid
  ForwardMapType forward_error_mappings_;

  /// Output value mapping grid
  ForwardMapType output_mappings_;

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
#include <ffnn/impl/layer/convolution.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_H
