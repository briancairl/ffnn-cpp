/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_ACTIVATION_CONFIGURATION_H
#define FFNN_LAYER_ACTIVATION_CONFIGURATION_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

namespace ffnn
{
namespace layer
{
namespace activation
{
/// Layer configuration struct
template<typename LayerType,
         typename ValueType,
         typename Options,
         typename Extrinsic>
class Configuration
{
public:
  friend LayerType;

  /// Shape type standardization
  typedef typename LayerType::ShapeType ShapeType;

  /**
   * @brief Default constructor
   */
  Configuration(size_type output_size = Options::output_size) :
    input_size_(output_size),
    output_size_(output_size)
  {}

  /**
   * @brief Sets layer input shape
   * @param height  height of the input volume
   * @param width   width of the input volume
   * @param depth   depth of the input volume
   * @return *this
   */
  inline Configuration& setInputShape(size_type height, size_type width = 1, size_type depth = 1)
  {
    FFNN_ASSERT_MSG(height > 0, "Input height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Input width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Input depth must be positive.");

    input_size_  = ShapeType(height, width, depth).size();
    output_size_ = input_size_;
    return *this;
  }

  /**
   * @brief Sets layer input shape
   * @param height  height of the input volume
   * @param width   width of the input volume
   * @param depth   depth of the input volume
   * @return *this
   */
  inline Configuration& setOutputShape(size_type height, size_type width = 1, size_type depth = 1)
  {
    FFNN_ASSERT_MSG(height > 0, "Output height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Output width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Output depth must be positive.");

    output_size_ = ShapeType(height, width, depth).size();
    input_size_  = output_size_;
    return *this;
  }

private:
  /// Number of layer inputs
  size_type input_size_;

  /// Number of layer outputs
  size_type output_size_;
};
}  // namespace activation
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_ACTIVATION_CONFIGURATION_H
