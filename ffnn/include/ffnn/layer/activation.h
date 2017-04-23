/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_ACTIVATION_H
#define FFNN_LAYER_ACTIVATION_H

// Boost
#include <boost/dynamic_bitset.hpp>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/neuron/neuron.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Activation layer
 */
template<typename ValueType,
         template<class> class NeuronType,
         FFNN_SIZE_TYPE SizeAtCompileTime = Eigen::Dynamic>
class Activation :
  public Hidden<ValueType, SizeAtCompileTime, SizeAtCompileTime>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, SizeAtCompileTime, SizeAtCompileTime>;

  /// Self type alias
  using Self = Activation<ValueType, NeuronType, SizeAtCompileTime>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Bia vector type standardization
  typedef typename Base::OutputVector BiasVector;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// A configuration object for a FullyConnected hidden layer
  struct Parameters
  {
    /// Standard deviation of biases on init
    ScalarType std_bias;

    /**
     * @brief Setup constructor
     * @param std_bias  Standard deviation of initial weights
     */
    Parameters(ScalarType std_bias = 1e-3);
  };

  /**
   * @brief Default constructor
   * @param config  layer configuration object
   */
  Activation(const Parameters& config = Parameters());
  virtual ~Activation();

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

protected:
  FFNN_REGISTER_SERIALIZABLE(Activation)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  FFNN_REGISTER_OPTIMIZER(Activation, GradientDescent);

  /// Layer configuration parameters
  Parameters config_;

  /// Bias vector
  BiasVector b_;

  /// Biased input
  BiasVector b_input_;

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
#include <ffnn/layer/impl/activation.hpp>
#endif  // FFNN_LAYER_ACTIVATION_H
