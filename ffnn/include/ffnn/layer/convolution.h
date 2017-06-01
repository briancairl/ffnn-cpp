/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_H
#define FFNN_LAYER_CONVOLUTION_H

// Boost
#include "boost/multi_array.hpp"

// FFNN
#include <ffnn/config/global.h>
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
 * @brief Describes compile-time traits used to set up a Convolution object
 * @param ValueType scalar value type
 * @param HeightAtCompileTime  height of the filter kernel
 * @param WidthAtCompileTime  width of the filter kernel
 * @param DepthAtCompileTime  depth of the filter kernel
 * @param KernelsAtCompileTime  number of kernels this filter will have
 */
template<typename ValueType,
         size_type HeightAtCompileTime = Eigen::Dynamic,
         size_type WidthAtCompileTime = Eigen::Dynamic,
         size_type DepthAtCompileTime = Eigen::Dynamic,
         size_type KernelHeightAtCompileTime = Eigen::Dynamic,
         size_type KernelWidthAtCompileTime = Eigen::Dynamic,
         size_type KernelCountAtCompileTime = Eigen::Dynamic,
         size_type RowStrideAtCompileTime =  1,
         size_type ColStrideAtCompileTime = -1,
         EmbeddingMode Mode = ColEmbedding>
struct convolution_layer_traits
{
  /// Data embedding mode
  constexpr static EmbeddingMode embedding_mode = Mode;

  /// Kernel kernel height
  constexpr static size_type kernel_height = KernelHeightAtCompileTime;

  /// Kernel kernel width
  constexpr static size_type kernel_width  = KernelWidthAtCompileTime;

  /// Number of filter kernels
  constexpr static size_type kernel_count  = KernelCountAtCompileTime;

  /// Filter stride along input rows
  constexpr static size_type row_stride  = RowStrideAtCompileTime;

  /// Filter stride along input cols
  constexpr static size_type col_stride  = ColStrideAtCompileTime;

  /// Input volume height
  constexpr static size_type input_height = HeightAtCompileTime;

  /// Input volume width
  constexpr static size_type input_width = WidthAtCompileTime;

  /// Input volume depth
  constexpr static size_type input_depth = DepthAtCompileTime;

  /// Output volume height
  constexpr static size_type output_height =
    convolution::output_dimension(HeightAtCompileTime, kernel_height, row_stride);

  /// Output volume width
  constexpr static size_type output_width =
    convolution::output_dimension(WidthAtCompileTime,  kernel_height, col_stride);

  /// Output volume depth
  constexpr static size_type output_depth = KernelCountAtCompileTime;

  /// Depth-embedded input height
  constexpr static size_type embedded_input_height =
    convolution::embed_dimension<Mode, ColEmbedding>(input_height, input_depth);

  /// Depth-embedded input width
  constexpr static size_type embedded_input_width =
    convolution::embed_dimension<Mode, RowEmbedding>(input_width,  input_depth);

  /// Depth-embedded output height
  constexpr static size_type embedded_output_height =
    convolution::embed_dimension<Mode, ColEmbedding>(output_height, output_depth);

  /// Depth-embedded output width
  constexpr static size_type embedded_output_width =
    convolution::embed_dimension<Mode, RowEmbedding>(output_width,  output_depth);

  /// Filter traits type standardization
  typedef filter_traits<
    ValueType,
    kernel_height,
    kernel_height,
    input_depth,
    kernel_count,
    embedding_mode
  > FilterTraits;

  /// Compile-time Hidden layer traits
  typedef hidden_layer_traits<
    embedded_input_height,
    embedded_input_width,
    embedded_output_height,
    embedded_output_width
  > HiddenLayerTraits

  /// Hidden layer (base type) standardization
  typedef Hidden<ValueType, HiddenLayerTraits> HiddenLayerType;
};

/**
 * @brief A convolution layer
 */
template <typename ValueType,
          typename LayerTraits = convolution_layer_traits<ValueType>>
class Convolution :
  public LayerTraits::HiddenLayerType
{
public:
  /// Self type alias
  using SelfType = Convolution<ValueType, LayerTraits>;

  /// Base type alias
  using BaseType = LayerTraits::HiddenLayerType;

  /// Dimension type standardization
  typedef typename LayerTraits::HiddenLayerType::ShapeType ShapeType;

  /// Filter parameters type standardization
  typedef Filter<ValueType, typename LayerTraits::FilterTraits> ParametersType;

  /// 2D-value mapping standardization
  typedef boost::multi_array<ValueType*, 2> ForwardValueMapType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<SelfType> OptimizerType;

  /// Dsitribution standardization
  typedef distribution::Distribution<ValueType> DistributionType;

  /// Layer configuration struct
  class Configuration
  {
  public:
    friend SelfType;

    /**
     * @brief Default constructor
     */
    Configuration() :
      input_shape_(LayerTraits::input_height, LayerTraits::input_width, LayerTraits::input_depth),
      filter_shape_(LayerTraits::kernel_height, LayerTraits::kernel_width, LayerTraits::kernel_count),
      row_stride_(LayerTraits::row_stride),
      col_stride_(LayerTraits::col_stride),
      parameter_distribution_(boost::make_shared<typename distribution::StandardNormal<ValueType>>())
      optimizer_(boost::make_shared<typename optimizer::None<SelfType>>())
    {
      /// Try to resolve from template arguments
      resolve();
    }

    /**
     * @brief Sets layer optimization resource
     * @param optimizer  layer optimizer
     * @return *this
     */
    inline Configuration& setOptimizer(const typename OptimizerType::Ptr& optimizer)
    {
      optimizer_ = optimizer;
      return *this;
    }

    /**
     * @brief Sets layer parameter initialization distribution
     * @param distribution  value distribution resource
     * @return *this
     */
    inline Configuration& setParameterDistribution(const typename DistributionType::Ptr& distribution)
    {
      parameter_distribution_ = distribution;
      return *this;
    }

    /**
     * @brief Sets layer input shape
     * @param input_shape   shape of the layer input
     * @return *this
     */
    inline Configuration& setInputShape(const ShapeType& input_shape)
    {
      input_shape_ = input_shape;
      return resolve();
    }

    /**
     * @brief Sets layer filter shape options
     * @param filter_shape  shape of the receptive field
     * \n
     *                      <b>Note: </b>
     *                      The shape specified here is <code>(H, W, N-Kernels)</code>. Each kernel actually has the
     *                      dimensions <code>(H, W, D)</code>, where <code>(D)</code> is the depth of the layer input
     * @return *this
     */
    inline Configuration& setFilterShape(const ShapeType& filter_shape)
    {
      filter_shape_ = filter_shape;
      return resolve();
    }

    /**
     * @brief Sets filter stride options
     * @param row_stride  filter stride along rows of input volume
     * @param col_stride  filter stride along rows of input volume
     * \n
     *                      <b>Note: </b>
     *                      If <code>col_stride</code> is not specified, it is defaulted to <code>row_stride</code>
     * @return *this
     */
    inline Configuration& setStride(size_type row_stride, size_type col_stride =-1)
    {
      row_stride_ = row_stride;
      col_stride_ = (col_stride <= 0) ? row_stride : col_stride;
      return resolve();
    }

  private:
    /**
     * @brief Recompute sizing config from user inputs
     * @return *this
     */
    Configuration& resolve()
    {
      // Set depth-embedded input shape
      embedded_input_shape_ = convolution::embed_shape_transform<Mode>(input_shape_);

      // Setup output shape before depth embdedding
      output_shape_.height = convolution::output_dimension(input_shape_.height, filter_shape_.height, row_stride_);
      output_shape_.width  = convolution::output_dimension(input_shape_.width,  filter_shape_.width,  col_stride_);
      output_shape_.depth  = filter_shape_.depth;

      // Set depth-embedded output shape
      embedded_output_shape_ = convolution::embed_shape_transform<Mode>(output_shape_);

      // Set stride shape
      stride_shape_.height = row_stride_;
      stride_shape_.width  = col_stride_;
      stride_shape_.depth  = input_shape_.depth;
      stride_shape_ = convolution::embed_shape_transform<Mode>(stride_shape_);
      return *this;
    }

    /// Shape of layer input
    ShapeType input_shape_;

    /// Shape of layer output
    ShapeType output_shape_;

    /// Filter stride along input rows
    size_type row_stride_;

    /// Filter stride along input cols
    size_type col_stride_;

    /// Shape of receptive fields
    ShapeType filter_shape_;

    /// Stride between receptive fields; considers data-embedding
    ShapeType stride_shape_;

    /// Embdedded shape of layer input
    ShapeType embedded_input_shape_;

    /// Embdedded shape of layer output
    ShapeType embedded_output_shape_;

    /**
     * @brief Distribution used to initializer layer coefficients
     * @note  This will be the <code>distribution::StandardNormal</code> by default
     */
    typename DistributionType::Ptr parameter_distribution_;

    /**
     * @brief Weight optimization resource
     * @note  This will be the <code>optimizer::None</code> by default
     */
    typename OptimizerType::Ptr optimizer_;
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

  /**
   * @brief Exposes layer parameters
   * @return Convolution filter parameters
   */
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

  /// Layer configurations
  Configuration config_;

  /** 
   * @brief Shape of the ouput
   * @note This is unlike BaseType::input_shape_ as it represents the layer input
   *       shape without regard for depth-embedding
   **/
  ShapeType input_volume_shape_;

  /**
   * @brief Shape of the ouput
   * @note This is unlike BaseType::input_shape_ as it represents the layer output
   *       shape without regard for depth-embedding
   */
  ShapeType output_volume_shape_;

  /**
   * @brief Layer parameters
   * @note  For the Convolution layer, theses are Filter coefficients
   */
  ParametersType parameters_;

  /// Forward error mapping grid
  ForwardValueMapType forward_error_mappings_;

  /// Output value mapping grid
  ForwardValueMapType output_mappings_;

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_shape_.size()</code>
   */
  offset_type connectToForwardLayer(const Layer<ValueType>& next, offset_type offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/convolution.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_H
