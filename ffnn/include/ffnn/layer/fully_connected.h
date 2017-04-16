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

  /// Class verision type alias
  using ClassVersionType = traits::Serializable::ClassVersionType;

  /// Matrix-type standardization
  typedef typename Hidden<ValueType>::InputVector InputVector;

  /// Matrix-type standardization
  typedef typename Hidden<ValueType>::OutputVector OutputVector;

  /// Size-type standardization
  typedef typename Hidden<ValueType>::SizeType SizeType;

  /// Offset-type standardization
  typedef typename Hidden<ValueType>::OffsetType OffsetType;

  /// Neuron type standardization
  typedef neuron::Neuron<ValueType> Neuron;

  /// Input-output weight matrix
  typedef Eigen::Matrix<ValueType, OutputsAtCompileTime, InputsAtCompileTime> WeightMatrix;

  /// A configuration struct for a FullyConnected Hidden
  struct Config
  {
    /// Learning rate used during weight updates
    ValueType learning_rate;

    /// Standard deviation of weights on init
    ValueType weight_init_variance;

    /// Default constructor
    Config() :
      learning_rate(1e-3),
      weight_init_variance(1e-3)
    {}
  };

  /**
   * @brief Setup constructor
   * @param output_dim  number of outputs from the Hidden
   * @param config  layer configuration struct
   */
  FullyConnected(SizeType output_dim = OutputsAtCompileTime,
                 const Config& config = Config());
  virtual ~FullyConnected();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward();

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  bool backward();

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  bool update();

  /**
   * @brief Saves object contents
   * @param[out] os  input stream
   * @param version  class version number
   * @retval true  if object was loaded successfully
   * @retval false  otherwise
   */
  bool save(std::ostream& os, ClassVersionType version);

  /**
   * @brief Loads object contents
   * @param[in] is  input stream
   * @param version  class version number
   * @retval true  if object was loaded successfully
   * @retval false  otherwise
   */
  bool load(std::istream& is, ClassVersionType version);

protected:
  /// Layer configurations
  Config config_;

  /// Output neurons
  std::vector<typename Neuron::Ptr> neurons_;

private:
  /**
   * @brief Sets up the layer after initialization
   * @retval true  if setup succeeded
   * @retval false  otherwise
   * @note Called after initialization sequence
   */
  bool setup();

  /// Weight matrix
  WeightMatrix w_;

  /// Weight matrix delta
  WeightMatrix w_delta_;

  /// Weighted input vector on last call to <code>forward</code>
  OutputVector w_input_;

  /// Input vector on last call to <code>forward</code>
  InputVector prev_input_;

  /// Archive save helper
  template<class Archive>
  void save(Archive& ar, ClassVersionType version) const;

  /// Archive load helper
  template<class Archive>
  void load(Archive& ar, ClassVersionType version);

  BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
