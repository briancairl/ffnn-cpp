/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_HIDDEN_HIDDEN_H
#define FFNN_LAYER_HIDDEN_HIDDEN_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/serializable.h>
#include <ffnn/internal/traits.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/hidden/compile_time_options.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A network hidden-layer object
 */
template<typename ValueType,
         typename Options    = hidden::options<>,
         typename Extrinsics = hidden::extrinsics<ValueType, Options>>
class Hidden :
  public Extrinsics::LayerType
{
  FFNN_ASSERT_DONT_MODIFY_EXTRINSICS(hidden);
public:
  /// Self type alias
  using SelfType = Hidden<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::LayerType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Input block type standardization
  typedef typename Extrinsics::InputBlockType InputBlockType;

  /// Output block type standardization
  typedef typename Extrinsics::OutputBlockType OutputBlockType;

  /// Input mapping type standardization
  typedef typename Extrinsics::InputMappingType InputMappingType;

  /// Output mapping type standardization
  typedef typename Extrinsics::OutputMappingType OutputMappingType;

  /**
   * @brief Setup constructor
   * @param input_height  height of the input surface
   * @param input_width  width of the input surface
   * @param output_height  height of the output surface
   * @param output_width  width of the output surface
   */
  explicit
  Hidden(const ShapeType& input_shape  = ShapeType(Options::input_height,  Options::input_width),
         const ShapeType& output_shape = ShapeType(Options::output_height, Options::output_width));
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
  virtual offset_type connectToForwardLayer(const Layer<ValueType>& next, offset_type offset);

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
protected:
  FFNN_REGISTER_SERIALIZABLE(Hidden);

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);
#endif
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/hidden/hidden.hpp>
#endif  // FFNN_LAYER_HIDDEN_HIDDEN_H
