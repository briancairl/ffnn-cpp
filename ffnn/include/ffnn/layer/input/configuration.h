/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INPUT_CONFIGURATION_H
#define FFNN_LAYER_INPUT_CONFIGURATION_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

namespace ffnn
{
namespace layer
{
namespace input
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
   * @param input_size  Network input size
   */
  Configuration(size_type input_size = Options::input_size) :
    input_size_(input_size)
  {}

  /**
   * @brief Sets network input shape
   * @param height  height of the input volume
   * @param width   width of the input volume
   * @param depth   depth of the input volume
   * @return *this
   */
  inline Configuration& setInputShape(size_type height, size_type width = 1, size_type depth = 1)
  {
    FFNN_ASSERT_MSG(height > 0, "Network input height must be positive.");
    FFNN_ASSERT_MSG(width > 0,  "Network input width must be positive.");
    FFNN_ASSERT_MSG(depth > 0,  "Network input depth must be positive.");

    input_size_ = ShapeType(height, width, depth).size();
    return *this;
  }

private:
  /// Number of layer inputs
  size_type input_size_;
};
}  // namespace input
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INPUT_CONFIGURATION_H
