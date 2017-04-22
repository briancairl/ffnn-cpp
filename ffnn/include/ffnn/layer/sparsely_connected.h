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
         template<class> class NeuronType,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic>
class SparselyConnected :
  public Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>
{
public:
  /// Base-type alias
  using Base = Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Self-type alias
  using Self = SparselyConnected<ValueType, NeuronType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Scalar-type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size-type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset-type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Matrix-type standardization
  typedef typename Base::InputVector InputVector;

  /// Matrix-type standardization
  typedef typename Base::OutputVector OutputVector;

  /// Input-output weight matrix
  typedef Eigen::SparseMatrix<ValueType> WeightMatrix;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// A configuration object for a SparselyConnected hidden layer
  struct Parameters
  {
    /// Standard deviation of weights on init
    ScalarType std_weight;

    /// Standard deviation of biases on init
    ScalarType std_bias;

    /// Porbability that a connection exists between any input/output pair
    ScalarType connection_probability;

    /**
     * @brief Setup constructor
     * @param std_weight  Standard deviation of initial weights
     * @param std_bias  Standard deviation of initial bias
     */
    Parameters(ScalarType std_weight = 1e-3,
               ScalarType std_bias = 1e-3,
               ScalarType connection_probability = 0.5);
  };

  /**
   * @brief Setup constructor
   * @param output_dim  number of outputs from the Hidden
   * @param config  layer configuration struct
   */
  SparselyConnected(SizeType output_dim = OutputsAtCompileTime,
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
  FFNN_REGISTER_OPTIMIZER(SparselyConnected, GradientDescent);

  /// Layer configuration parameters
  Parameters config_;

  /// Weight matrix
  WeightMatrix w_;

  /// Bias vector
  OutputVector b_;

  /// Weighted input vector on last call to <code>forward</code>
  OutputVector w_input_;

  /// Layer activation units
  std::vector<NeuronType<ValueType>> neurons_;

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
