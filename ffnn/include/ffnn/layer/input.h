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

namespace ffnn
{
namespace layer
{
/**
 * @brief A layer which handles network inputs
 */
template<typename ValueType, FFNN_SIZE_TYPE NetworkInputsAtCompileTime = Eigen::Dynamic>
class Input :
  public Layer<ValueType>
{
public:
  /// Base-type alias
  using Base = Layer<ValueType>;

  /// Size-type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset-type standardization
  typedef typename Base::OffsetType OffsetType;

  /**
   * @brief Setup constructor
   * @param input_dim  number of inputs supplied to network by this Layer
   */
  Input(const SizeType& network_input_dim = NetworkInputsAtCompileTime);
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
   * @warning This method does not check element-type correctness
   */
  template<typename NetworkInputType>
  void operator<<(const NetworkInputType& input) const;

private:
  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval output_dimension_
   */
  OffsetType connectToForwardLayer(const Base& next, OffsetType offset);

  /// Pointer to first element of next layer
  ValueType* next_ptr_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/input.hpp>
#endif  // FFNN_LAYER_INPUT_H
