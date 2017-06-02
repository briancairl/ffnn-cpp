/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INPUT_H
#define FFNN_LAYER_INPUT_H

// C++ Standard Library
#include <cstring>
#include <iostream>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/shape.h>

namespace ffnn
{
namespace layer
{
namespace input
{
/**
 * @brief Describes compile-time options used to set up a Input object
 */
template<size_type InputsAtCompileTime = Eigen::Dynamic>
struct options
{
  /// Total network input size
  constexpr static size_type size = InputsAtCompileTime;
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  ///Layer (base type) standardization
  typedef Layer<ValueType> LayerType;
};
}  // namespace input

/**
 * @brief A layer which handles network inputs
 */
template<typename ValueType,
         typename Options    = input::options<>,
         typename Extrinsics = input::extrinsics<ValueType, Options>>
class Input :
  public Extrinsics::LayerType
{
  FFNN_ASSERT_NO_MOD_LAYER_EXTRINSICS(input);
{
public:
  /// Self type alias
  using SelfType = Input<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::LayerType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /**
   * @brief Setup constructor
   * @param input_size  number of inputs supplied to network by this Layer
   */
  explicit
  Input(size_type network_input_size = Options::size);
  virtual ~Input();

  /**
   * @brief Initialize the layer
   */
  bool initialize();

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
  offset_type connectToForwardLayer(const Base& next, offset_type offset);

  /// Pointer to first element of next layer
  ValueType* next_ptr_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/input.hpp>
#endif  // FFNN_LAYER_INPUT_H
