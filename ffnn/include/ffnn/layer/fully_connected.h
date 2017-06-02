/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN (internal)
#include <ffnn/layer/internal/shape.h>
#include <ffnn/internal/traits.h>

// FFNN
#include <ffnn/layer/hidden.h>
#include <ffnn/neuron/neuron.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

namespace ffnn
{
namespace layer
{
namespace fully_connected
{
/**
 * @brief Describes compile-time options used to set up a Input object
 */
template<size_type InputsAtCompileTime = Eigen::Dynamic,
         size_type InputWidthAtCompileTime  = 1,
         size_type InputDepthAtCompileTime  = 1>
struct options
{
  /// Input field height
  constexpr static size_type height = InputHeightAtCompileTime;

  /// Input field width
  constexpr static size_type width = InputWidthAtCompileTime;

  /// Input field depth
  constexpr static size_type depth = InputDepthAtCompileTime;

  /// Total network fully_connected size
  constexpr static size_type size =
    multiply_if_not_dynamic_sizes(height, width, depth);
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
}  // namespace fully_connected

/**
 * @brief A fully-connected layer
 */
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic,
         typename _HiddenLayerShape = hidden_layer_traits<InputsAtCompileTime, 1, OutputsAtCompileTime, 1>>
class FullyConnected :
  public Hidden<ValueType, _HiddenLayerShape>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, _HiddenLayerShape>;

  /// Self type alias
  using Self = FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Dimension type standardization
  typedef typename Base::ShapeType ShapeType;

  /// Matrix type standardization
  typedef typename Base::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename Base::OutputBlockType OutputBlockType;

  /// Bia vector type standardization
  typedef typename Base::OutputBlockType BiasVectorType;

  /// Input-output weight matrix
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime, Eigen::ColMajor> WeightMatrixType;

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
   * @brief Initialize layer weights and biases according to particular distributions
   * @param wd  distribution to sample for connection weights
   * @param bd  distribution to sample for biases
   * @retval true  if layer was initialized successfully
   * @retval false otherwise
   *
   * @warning If layer is a loaded instance, this method will initialize layer sizings
   *          but weights will not be reset according to the given distributions
   */
  template<typename WeightDistribution, typename BiasDistribution>
  bool initialize(const WeightDistribution& wd, const BiasDistribution& bd);

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

protected:
  FFNN_REGISTER_SERIALIZABLE(FullyConnected)

  /// Save serialize
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serialize
  void load(InputArchive& ar, VersionType version);

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
