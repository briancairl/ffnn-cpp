/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OUTPUT_OUTPUT_H
#define FFNN_LAYER_OUTPUT_OUTPUT_H

// C++ Standard Library
#include <cstring>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/output/compile_time_options.h>
#include <ffnn/layer/output/configuration.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A layer which handles network outputs
 */
template<typename ValueType,
         typename Options    = output::options<>,
         typename Extrinsics = output::extrinsics<ValueType, Options>>
class Output :
  public Extrinsics::LayerType
{
  FFNN_ASSERT_DONT_MODIFY_EXTRINSICS(output);
public:
  /// Self type alias
  using SelfType = Output<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::LayerType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Configuration type standardization
  typedef output::Configuration<SelfType, ValueType, Options, Extrinsics> Configuration;

  /**
   * @brief Setup constructor
   * @param config  layer configuration
   */
  explicit
  Output(const Configuration& config = Configuration());
  virtual ~Output();

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
   * @brief Get network output value
   * @param[out] output  network output data
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
  offset_type connectToForwardLayer(const BaseType& next, offset_type offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/output/output.hpp>
#endif  // FFNN_LAYER_OUTPUT_OUTPUT_H
