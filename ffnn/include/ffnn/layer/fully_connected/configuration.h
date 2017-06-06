/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_CONFIGURATION_H
#define FFNN_LAYER_FULLY_CONNECTED_CONFIGURATION_H

// Boost
#include <boost/make_shared.hpp>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

#include <ffnn/distribution/distribution.h>
#include <ffnn/distribution/normal.h>

#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace layer
{
namespace fully_connected
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

  /// Distribution type standardization
  typedef distribution::Distribution<ValueType> DistributionType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<LayerType> OptimizerType;

  /**
   * @brief Default constructor
   */
  Configuration(size_type output_size = Options::output_size) :
    input_size_(Options::input_size),
    output_size_(output_size),
    distribution_(boost::make_shared<typename distribution::StandardNormal<ValueType>>()),
    optimizer_(boost::make_shared<typename optimizer::None<LayerType>>())
  {}

  /**
   * @brief Sets layer optimization resource
   * @param optimizer  layer optimizer
   * @return *this
   */
  inline Configuration& setOptimizer(const typename OptimizerType::Ptr& optimizer)
  {
    FFNN_ASSERT_MSG(optimizer, "Optimizer resource is invalid.");
    FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected::Configuration",
                              "Set option: [" << optimizer->name() << "] optimizer.");
    optimizer_ = optimizer;
    return *this;
  }

  /**
   * @brief Sets layer parameter initialization distribution
   * @param distribution  value distribution resource
   * @return *this
   */
  inline Configuration& setParameterDistribution(const typename DistributionType::Ptr& distribution)
  {
    FFNN_ASSERT_MSG(distribution, "Distribution resource is invalid.");
    FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected::Configuration",
                              "Set parameter initialization distribution.");
    distribution_ = distribution;
    return *this;
  }

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

    input_size_ = ShapeType(height, width, depth).size();
    FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected::Configuration",
                              "Set option: [" << input_size_ << "] as total input count.");
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
    FFNN_INTERNAL_DEBUG_NAMED("layer::FullyConnected::Configuration",
                              "Set option: [" << output_size_ << "] as total output count.");
    return *this;
  }

private:
  /// Number of layer inputs
  size_type input_size_;

  /// Number of layer outputs
  size_type output_size_;

  /**
   * @brief Distribution used to initializer layer coefficients
   * @note  This will be the <code>distribution::StandardNormal</code> by default
   */
  typename DistributionType::Ptr distribution_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> by default
   */
  typename OptimizerType::Ptr optimizer_;

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
  #include <ffnn/impl/layer/fully_connected/configuration/serialization_class_definitions.hpp>
#endif
};
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_FULLY_CONNECTED_CONFIGURATION_H
