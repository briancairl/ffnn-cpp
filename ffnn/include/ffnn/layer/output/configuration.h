/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OUTPUT_CONFIGURATION_H
#define FFNN_LAYER_OUTPUT_CONFIGURATION_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

namespace ffnn
{
namespace layer
{
namespace output
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
   * @param output_size  Network output size
   */
  Configuration(size_type output_size = Options::output_size) :
    output_size_(output_size)
  {}

  /**
   * @brief Sets layer output shape
   * @param height  height of the output volume
   * @param width   width of the output volume
   * @param depth   depth of the output volume
   * @return *this
   */
  inline Configuration& setOutputShape(size_type height, size_type width = 1, size_type depth = 1)
  {
    FFNN_ASSERT_MSG(height > 0, "Network output height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Network output width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Network output depth must be positive.");

    output_size_ = ShapeType(height, width, depth).size();
    return *this;
  }

private:
  /// Number of layer outputs
  size_type output_size_;
};
}  // namespace output
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_OUTPUT_CONFIGURATION_H
