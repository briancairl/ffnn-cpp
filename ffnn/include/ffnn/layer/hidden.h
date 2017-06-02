/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H
#define FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/layer/layer.h>
#include <ffnn/internal/serializable.h>
#include <ffnn/internal/traits.h>

namespace ffnn
{
namespace layer
{
namespace hidden
{
/**
 * @brief Describes compile-time options used to set up a Convolution object
 */
template<size_type InputHeightAtCompileTime  = Eigen::Dynamic,
         size_type InputWidthAtCompileTime   = Eigen::Dynamic,
         size_type OutputHeightAtCompileTime = Eigen::Dynamic,
         size_type OutputWidthAtCompileTime  = Eigen::Dynamic,
         int InputDataOrdering  = Eigen::ColMajor,
         int OutputDataOrdering = Eigen::ColMajor>
struct options
{
  /// Input field height
  constexpr static size_type input_height  = InputHeightAtCompileTime;

  /// Input field width
  constexpr static size_type input_width = InputWidthAtCompileTime;

  /// Input data ordering
  constexpr static int input_data_ordering = InputDataOrdering;

  /// Output field height
  constexpr static size_type output_height = OutputHeightAtCompileTime;

  /// Output field width
  constexpr static size_type output_width = OutputWidthAtCompileTime;

  /// Output data ordering
  constexpr static int output_data_ordering = OutputDataOrdering;
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Input block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::input_height,
    Options::input_width,
    Options::input_data_ordering
  > InputBlockType;

  /// Output block type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::output_height,
    Options::output_width,
    Options::output_data_ordering
  > OutputBlockType;

  /// Input block-mapping type standardization
  typedef typename std::conditional<
    std::is_floating_point<ValueType>::value,
    Eigen::Map<InputBlockType, 16>,
    Eigen::Map<InputBlockType>
  >::type InputMappingType;

  /// Output block-mapping type standardization
  typedef typename std::conditional<
    std::is_floating_point<ValueType>::value,
    Eigen::Map<OutputBlockType, 16>,
    Eigen::Map<OutputBlockType>
  >::type OutputMappingType;

  ///Layer (base type) standardization
  typedef Layer<ValueType> LayerType;
};
}  // namespace hidden

/**
 * @brief A network hidden-layer object
 */
template<typename ValueType,
         typename Options = hidden::options<>,
         typename Extrinsics = hidden::extrinsics<ValueType, Options>>
class Hidden :
  public Extrinsics::LayerType
{
public:
  /// Self type alias
  using SelfType = Hidden<ValueType, Options>;

  /// Base type alias
  using BaseType = Layer<ValueType>;

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
  virtual offset_type connectToForwardLayer(const Layer<ValueType>& next, offset_type offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/hidden.hpp>
#endif  // FFNN_LAYER_INTERNAL_HIDDEN_INTERFACE_H
