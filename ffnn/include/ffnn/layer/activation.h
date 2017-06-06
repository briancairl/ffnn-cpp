/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_ACTIVATION_ACTIVATION_H
#define FFNN_LAYER_ACTIVATION_ACTIVATION_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/internal/config.h>
#include <ffnn/layer/activation/configuration.h>
#include <ffnn/layer/activation/compile_time_options.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Activation layer
 */
template<typename ValueType,
         typename NeuronType,
         typename Options    = activation::options<>,
         typename Extrinsics = activation::extrinsics<ValueType, NeuronType, Options>>
class Activation :
  public Extrinsics::HiddenLayerType
{
  FFNN_ASSERT_DONT_MODIFY_EXTRINSICS_EXT(activation, NeuronType);
public:
  /// Self type alias
  using SelfType = Activation<ValueType, NeuronType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::HiddenLayerType;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Neuron block type standardization
  typedef typename Extrinsics::NeuronBlockType NeuronBlockType;

  /// Configuration type standardization
  typedef activation::Configuration<SelfType, ValueType, Options, Extrinsics> Configuration;

  /**
   * @brief Setup constructor
   * @param config  Layer configuration struct
   */
  explicit Activation(const Configuration& config = Configuration());
  virtual ~Activation();

  /**
   * @brief Initialize the layer
   */
  bool initialize();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  bool backward();

  /**
   * @brief Does nothing
   * @retval true  always
   * @note The activation layer has no parameters to update
   */
  bool update()
  {
    return true;
  }

  /**
   * @brief Reset Neuron block
   */
  template<bool T = Options::has_fixed_sizes>
  typename std::enable_if<T>::type reset();
  template<bool T = Options::has_fixed_sizes>
  typename std::enable_if<!T>::type reset();

private:
  /// Layer configurations
  Configuration config_;

  /// Layer activation units
  NeuronBlockType neurons_;

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
protected:
  FFNN_REGISTER_SERIALIZABLE(Activation);

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);
#endif
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/activation/activation.hpp>
#endif  // FFNN_LAYER_ACTIVATION_ACTIVATION_H
