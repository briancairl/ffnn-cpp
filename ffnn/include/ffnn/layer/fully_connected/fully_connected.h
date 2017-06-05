/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/internal/traits.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/fully_connected/configuration.h>
#include <ffnn/layer/fully_connected/compile_time_options.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A fully-connected layer
 */
template <typename ValueType,
          typename Options    = typename fully_connected::options<>,
          typename Extrinsics = typename fully_connected::extrinsics<ValueType, Options>>
class FullyConnected :
  public Extrinsics::HiddenLayerType
{
public:
  /// Self type alias
  using SelfType = FullyConnected<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::HiddenLayerType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Parameters (connection weights) type standardization
  typedef typename Extrinsics::ParametersType ParametersType;

  /// Configuration type standardization
  typedef fully_connected::Configuration<SelfType, ValueType, Options, Extrinsics> Configuration;

  /**
   * @brief Setup constructor
   * @param config  Layer configuration struct
   */
  explicit FullyConnected(const Configuration& config = Configuration());
  virtual ~FullyConnected();

  /**
   * @brief Initialize the layer
   * @retval true  if layer was initialized successfully
   * @retval false otherwise
   *
   * @warning If layer is not loaded instance, this method will initialize layer sizings
   *          but weights and biases will be zero
   */
  bool initialize();

  /**
   * @brief Performs forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   * @warning Does not apply layer weight updates
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  bool backward();

  /**
   * @brief Applies accumulated layer weight updates computed during optimization
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  bool update();

  /**
   * @brief Reset weights and biases
   */
  void reset();

  /**
   * @brief Exposes layer parameters
   * @return Connection weight parameters
   */
  inline const ParametersType& getParameters() const
  {
    return parameters_;
  }

private:
  FFNN_REGISTER_OPTIMIZER(FullyConnected, Adam);
  FFNN_REGISTER_OPTIMIZER(FullyConnected, GradientDescent);

  /// Layer configurations
  Configuration config_;

  /**
   * @brief Layer parameters
   * @note  For the FullyConnected layer, theses are connection Weights
   */
  ParametersType parameters_;

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
protected:
  FFNN_REGISTER_SERIALIZABLE(FullyConnected);

  /// Save serialize
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serialize
  void load(InputArchive& ar, VersionType version);
#endif
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/fully_connected/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_FULLY_CONNECTED_H
