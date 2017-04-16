/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// C++ Standard Library
#include <iostream>
#include <string>
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
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic>
class FullyConnected :
  public Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>
{
public:
  /// Base-type alias
  using Base = Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>;

  /// Sekf-type alias
  using Self = FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>;

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
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime> WeightMatrix;

  /// Neuron type standardization
  typedef neuron::Neuron<ValueType> Neuron;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// A configuration object for a FullyConnected hidden layer
  struct Parameters
  {
    /// Standard deviation of weights on init
    ScalarType std_weight;

    /// Standard deviation of biases on init
    ScalarType std_bias;

    /**
     * @brief Setup constructor
     * @param std_weight  Standard deviation of initial weights
     * @param std_bias  Standard deviation of initial bias
     */
    Parameters(ScalarType std_weight = 1e-3, ScalarType std_bias = 1e-3);
  };

  /**
   * @brief Setup constructor
   * @param output_dim  number of outputs from the Hidden
   * @param config  layer configuration struct
   */
  FullyConnected(SizeType output_dim = OutputsAtCompileTime,
                 const Parameters& config = Parameters());
  virtual ~FullyConnected();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward();

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool backward();

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  virtual bool update();

  /**
   * @brief Reset weights and biases
   */
  void reset();

  /**
   * @brief Sets an optimizer used update network weights during back-propagation
   * @param opt  optimizer to set
   */
  void setOptimizer(typename Optimizer::Ptr opt);

protected:
  FFNN_REGISTER_SERIALIZABLE(FullyConnected)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  FFNN_REGISTER_OPTIMIZER(FullyConnected, GradientDescent);

  /// Layer configuration
  Parameters config_;

  /// Weight matrix
  WeightMatrix w_;

  /// Bias vector
  OutputVector b_;

  /// Weighted input vector on last call to <code>forward</code>
  OutputVector w_input_;

  /// Output neurons
  std::vector<typename Neuron::Ptr> neurons_;

  /// Weight optimization resource
  typename Optimizer::Ptr opt_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
