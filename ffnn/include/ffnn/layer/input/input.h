/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INPUT_INPUT_H
#define FFNN_LAYER_INPUT_INPUT_H

// C++ Standard Library
#include <cstring>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/input/compile_time_options.h>
#include <ffnn/layer/input/configuration.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A layer which handles network inputs
 */
template<typename ValueType,
         typename Options    = input::options<>,
         typename Extrinsics = input::extrinsics<ValueType, Options>>
class Input :
  public Extrinsics::LayerType
{
  FFNN_ASSERT_DONT_MODIFY_EXTRINSICS(input);
public:
  /// Self type alias
  using SelfType = Input<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::LayerType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Configuration type standardization
  typedef input::Configuration<SelfType, ValueType, Options, Extrinsics> Configuration;

  /**
   * @brief Setup constructor
   * @param config  layer configuration
   */
  explicit
  Input(const Configuration& config = Configuration());
  virtual ~Input();

  /**
   * @brief Initialize the layer
   */
  bool initialize();

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  bool update()
  {
    return true;
  };

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward()
  {
    return true;
  };

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  bool backward()
  {
    return true;
  };

  /**
   * @brief Sets network input values
   * @param input  network input data
   * @note <code>NetworkInputType</code> must have the following methods
   *       - <code>NetworkInputType::data()</code> to expose a pointer to a contiguous memory block
   *       - <code>NetworkInputType::size()</code> to expose the size of the memory block
   * @warning This method does not check element type correctness
   */
  template<typename NetworkInputType>
  void operator<<(const NetworkInputType& input) const;

private:

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval output_shape_.size()
   */
  offset_type connectToForwardLayer(const BaseType& next, offset_type offset);

  /// Layer configuration struct
  Configuration config_;

  /// Pointer to first element of next layer
  ValueType* next_ptr_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/input/input.hpp>
#endif  // FFNN_LAYER_INPUT_INPUT_H
