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
#include <ffnn/layer/hidden.h>
#include <ffnn/neuron/neuron.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

namespace ffnn
{
namespace layer
{

template<class BlockType>
struct is_alignable_128 :
  std::conditional<
    (sizeof(BlockType)%16) == 0,
    std::true_type,
    std::false_type
  >::type
{};

template <typename ConnectType, typename BiasType>
class LayerParameters
{
  /// Require that ScalarType is same between Connecting/Biasing types
  static_assert(std::is_same<typename ConnectType::Scalar, typename BiasType::Scalar>::value,
                "Scalar representation used by [ConnectionType] and [BiasType] must match.");

  /// Require that ScalarType is floating point
  static_assert(std::is_floating_point<typename ConnectType::Scalar>::value,
                "Scalar representation must be a floating point type.");
public:
  /**
   * @brief Scalar-type standardization
   * @warning  Scalar-type consistency between ConnectType and BiasType is enforced
   * @warning  Scalar-type enforced as floating-point type
   */
  typedef typename ConnectType::Scalar ScalarType;

  /**
   * @brief An initialization configuration object used for setting up a LayerParameters object
   */
  struct InitConfiguration
  {
    /// Standard deviation of connection weights on init
    ScalarType init_connection_std;

    /// Standard deviation of biases on init
    ScalarType init_bias_std;

    /// Connection weight mean (bias) on init
    ScalarType init_connection_mean;

    /// Connection biasing mean (bias) on init
    ScalarType init_bias_mean;

    /**
     * @brief Setup constructor
     * @param init_connection_std  Standard deviation of initial weights
     * @param init_bias_std  Standard deviation of initial weights
     * @param init_connection_mean  Mean of intial weights
     * @param init_bias_mean  Mean of intial biases
     */
    explicit
    InitConfiguration(ScalarType init_connection_std = 1e-3,
                      ScalarType init_bias_std = 1e-3,
                      ScalarType init_connection_mean = 0.0,
                      ScalarType init_bias_mean = 0.0);
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(is_alignable_128<ConnectType>::value ||
                                     is_alignable_128<BiasType>::value);
};

/**
 * @brief A fully-connected layer
 */
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic,
         typename _HiddenLayerShape = hidden_layer_shape<InputsAtCompileTime, 1, OutputsAtCompileTime, 1>>
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

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

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

  /// Layer parameter type
  typedef LayerParameters<WeightMatrixType, BiasVectorType> LayerParameterType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// A configuration object for a FullyConnected hidden layer
  struct Parameters
  {
    /// Standard deviation of connection weights on init
    ScalarType init_weight_std;

    /// Standard deviation of biases on init
    ScalarType init_bias_std;

    /// Connection weight mean (bias) on init
    ScalarType init_weight_mean;

    /// Connection biasing mean (bias) on init
    ScalarType init_bias_mean;

    /**
     * @brief Setup constructor
     * @param init_weight_std  Standard deviation of initial weights
     * @param init_bias_std  Standard deviation of initial weights
     * @param init_weight_mean  Mean of intial weights
     * @param init_bias_mean  Mean of intial biases
     */
    explicit
    Parameters(ScalarType init_weight_std = 1e-3,
               ScalarType init_bias_std = 1e-3,
               ScalarType init_weight_mean = 0.0,
               ScalarType init_bias_mean = 0.0);
  };

  /**
   * @brief Setup constructor
   * @param output_size  number of layer outputs
   * @param config  layer configuration struct
   */
  explicit
  FullyConnected(SizeType output_size = OutputsAtCompileTime,
                 const Parameters& config = Parameters());
  virtual ~FullyConnected();

  /**
   * @brief Initialize the layer
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

protected:
  FFNN_REGISTER_SERIALIZABLE(FullyConnected)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  FFNN_REGISTER_OPTIMIZER(FullyConnected, Adam);
  FFNN_REGISTER_OPTIMIZER(FullyConnected, GradientDescent);

  /// Layer configuration parameters
  Parameters config_;

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
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
