/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H
#define FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN (internal)
#include <ffnn/internal/serializable.h>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/layer/layer.h>

namespace ffnn
{
namespace layer
{
template<FFNN_SIZE_TYPE InputHeightAtCompileTime  = Eigen::Dynamic,
         FFNN_SIZE_TYPE InputWidthAtCompileTime   = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputHeightAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputWidthAtCompileTime  = Eigen::Dynamic>
struct hidden_layer_shape
{
  constexpr static FFNN_SIZE_TYPE input_height  = InputHeightAtCompileTime;
  constexpr static FFNN_SIZE_TYPE input_width   = InputWidthAtCompileTime;
  constexpr static FFNN_SIZE_TYPE output_height = OutputHeightAtCompileTime;
  constexpr static FFNN_SIZE_TYPE output_width  = OutputWidthAtCompileTime;
};

/**
 * @brief A network hidden-layer object
 */
template<typename ValueType,
         typename LayerShape = hidden_layer_shape<>,
         typename _InputBlockType  = Eigen::Matrix<ValueType, LayerShape::input_height,  LayerShape::input_width,  Eigen::ColMajor>,
         typename _OutputBlockType = Eigen::Matrix<ValueType, LayerShape::output_height, LayerShape::output_width, Eigen::ColMajor>>
class Hidden :
  public Layer<ValueType>
{
public:
  /// Base type alias
  using Base = Layer<ValueType>;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Input block type standardization
  typedef _InputBlockType InputBlockType;

  /// Output block type standardization
  typedef _OutputBlockType OutputBlockType;

  /// Input block-mapping type standardization
  typedef Eigen::Map<InputBlockType, 16> InputMappingType;

  /// Output block-mapping type standardization
  typedef Eigen::Map<OutputBlockType, 16> OutputMappingType;

  /**
   * @brief Setup constructor
   * @param input_height  height of the input surface
   * @param input_width  width of the input surface
   * @param output_height  height of the output surface
   * @param output_width  width of the output surface
   */
  explicit
  Hidden(const ShapeType& input_shape  = ShapeType(LayerShape::input_height,  LayerShape::input_width),
         const ShapeType& output_shape = ShapeType(LayerShape::output_height, LayerShape::output_width));
  virtual ~Hidden();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward() = 0;

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool backward() = 0;

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  virtual bool update() = 0;

protected:
  FFNN_REGISTER_SERIALIZABLE(Hidden)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

  /// Memory-mapped input vector
  InputMappingType input_;

  /// Memory-mapped output vector
  OutputMappingType output_;

  /// Backward error vector
  InputMappingType backward_error_;

  /// Output-target error vector
  OutputMappingType forward_error_;

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_shape_.size()</code>
   */
  virtual OffsetType connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/hidden.hpp>
#endif  // FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H
