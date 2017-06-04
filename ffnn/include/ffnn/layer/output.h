/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OUTPUT_H
#define FFNN_LAYER_OUTPUT_H

// C++ Standard Library
#include <iostream>

// FFNN
#include <ffnn/internal/config.h>
#include <ffnn/layer/layer.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A layer which handles network outputs
 */
template<typename ValueType,
         FFNN_SIZE_TYPE NetworkOutputsAtCompileTime = Eigen::Dynamic>
class Output :
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

  /**
   * @brief Default constructor
   */
  Output();
  virtual ~Output();

  /**
   * @brief Initialize the layer
   */
  bool initialize();

  /**
   * @brief Get network output value
   * @param[out] output  network input data
   * @note <code>NetworkOutputType</code> must have the following methods
   *       - <code>NetworkOutputType::data()</code> to expose a pointer to a contiguous memory block
   *       - <code>NetworkOutputType::size()</code> to expose the size of the memory block
   * @warning This method does not check element type correctness
   */
  template<typename NetworkOutputType>
  void operator>>(NetworkOutputType& output);

  /**
   * @brief Set network output-target value
   * @param target  network output-target values
   * @note <code>NetworkTargetType</code> must have the following methods
   *       - <code>NetworkTargetType::data()</code> to expose a pointer to a contiguous memory block
   *       - <code>NetworkTargetType::size()</code> to expose the size of the memory block
   * @warning This method does not check element type correctness
   */
  template<typename NetworkTargetType>
  void operator<<(const NetworkTargetType& target);

private:
  /**
   * @brief Passthrough
   * @note  The Output layer is the terminal layer of a network
   */
  OffsetType connectToForwardLayer(const Base& next, OffsetType offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/output.hpp>
#endif  // FFNN_LAYER_OUTPUT_H
