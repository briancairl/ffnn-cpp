/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_SPARSELY_CONNECTED_H
#define FFNN_LAYER_SPARSELY_CONNECTED_H

// C++ Standard Library
#include <vector>

// FFNN
#include <ffnn/layer/hidden.h>
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
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic,
         typename _HiddenLayerShape = hidden_layer_shape<InputsAtCompileTime, 1, OutputsAtCompileTime, 1>>
class SparselyConnected :
  public Hidden<ValueType, _HiddenLayerShape>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, _HiddenLayerShape>;

  /// Self type alias
  using Self = SparselyConnected<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

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
  typedef Eigen::SparseMatrix<ValueType, Eigen::ColMajor> WeightMatrixType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// A configuration object for a SparselyConnected hidden layer
  struct Parameters
  {
    /// Probability that a connection exists between any input/output pair
    ScalarType connection_probability;

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
     * @param connection_probability  Probability a connection will exist between any input/output pair
     * @param init_weight_std  Standard deviation of initial weights
     * @param init_bias_std  Standard deviation of initial weights
     * @param init_weight_mean  Mean of intial weights
     * @param init_bias_mean  Mean of intial biases
     */
    explicit
    Parameters(ScalarType connection_probability = 0.5,
               ScalarType init_weight_std = 1e-3,
               ScalarType init_bias_std = 1e-3,
               ScalarType init_weight_mean = 0.0,
               ScalarType init_bias_mean = 0.0);
  };

  /**
   * @brief Setup constructor
   * @param output_size  number of outputs from the layer
   * @param config  layer configuration struct
   */
  explicit
  SparselyConnected(SizeType output_size = OutputsAtCompileTime,
                    const Parameters& config = Parameters());
  virtual ~SparselyConnected();

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
   * @brief Prune weights smaller than some value epsilon
   * @param epislon  minimum bounding value for weights to keep after pruning 
   */
  void prune(ValueType epsilon);

  /**
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   * @warning <code>backward</code> and <code>update</code> methods are expected to throw if an
   *          optimizer has not been set explicitly
   */
  void setOptimizer(typename Optimizer::Ptr opt);

protected:
  FFNN_REGISTER_SERIALIZABLE(SparselyConnected)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  FFNN_REGISTER_OPTIMIZER(SparselyConnected, Adam);
  FFNN_REGISTER_OPTIMIZER(SparselyConnected, GradientDescent);

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
#include <ffnn/layer/impl/sparsely_connected.hpp>
#endif  // FFNN_LAYER_SPARSELY_CONNECTED_H
