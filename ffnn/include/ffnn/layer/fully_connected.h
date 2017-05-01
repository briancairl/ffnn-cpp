/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// C++ Standard Library
#include <vector>

// FFNN
#include <ffnn/layer/internal/hidden_interface.h>
#include <ffnn/neuron/neuron.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A fully-connected layer
 */
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic>
class FullyConnected :
  public HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime>
{
public:
  /// Base type alias
  using Base = HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Self type alias
  using Self = FullyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Matrix type standardization
  typedef typename Base::InputVectorType InputVectorType;

  /// Matrix type standardization
  typedef typename Base::OutputVectorType OutputVectorType;

  /// Bia vector type standardization
  typedef typename Base::OutputVectorType BiasVector;

  /// Input-output weight matrix
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime, Eigen::ColMajor> WeightMatrix;

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
  virtual bool initialize();

  /**
   * @brief Performs forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   * @warning Does not apply layer weight updates
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  virtual bool backward();

  /**
   * @brief Applies accumulated layer weight updates computed during optimization
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  virtual bool update();

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
  inline const WeightMatrix& getWeights() const
  {
    return w_;
  }

  /**
   * @brief Exposes internal biasing weights
   * @return input-biasing vector
   */
  inline const BiasVector& getBiases() const
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
  WeightMatrix w_;

  /// Bias vector
  BiasVector b_;

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
