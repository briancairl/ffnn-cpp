/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/layer/internal/shape.h>
#include <ffnn/internal/traits.h>

#include <ffnn/layer/hidden.h>
#include <ffnn/layer/fully_connected/compile_time_options.h>

#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

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
  typedef typename Base::ShapeType ShapeType;

  /// Parameters (connection weights) type standardization
  typedef Extrinsics::ConectionWeightsType ParametersType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /**
   * @brief Setup constructor
   * @param output_size  number of layer outputs
   */
  explicit FullyConnected(size_type output_size = OutputsAtCompileTime);
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
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   * @warning <code>backward</code> and <code>update</code> methods are expected to throw if an
   *          optimizer has not been set explicitly
   */
  void setOptimizer(typename Optimizer::Ptr opt);

  /**
   * @brief Exposes internal connection weights
   * @return input-output connection weights
   */
  inline const WeightMatrixType& getWeights() const
  {
    return w_;
  }

  /**
   * @brief Exposes internal biasing weights
   * @return input-biasing vector
   */
  inline const BiasVectorType& getBiases() const
  {
    return b_;
  }

private:
  FFNN_REGISTER_OPTIMIZER(FullyConnected, Adam);
  FFNN_REGISTER_OPTIMIZER(FullyConnected, GradientDescent);

  /// Weight matrix
  WeightMatrixType w_;

  /// Bias vector
  BiasVectorType b_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> type by default
   * @see   setOptimizer
   */
  typename Optimizer::Ptr opt_;

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
protected:
  FFNN_REGISTER_SERIALIZABLE(FullyConnected)

  /// Save serialize
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serialize
  void load(InputArchive& ar, VersionType version);
#endif

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(
    ffnn::internal::traits::is_alignable_128<WeightMatrixType>::value ||
    ffnn::internal::traits::is_alignable_128<BiasVectorType>::value);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
