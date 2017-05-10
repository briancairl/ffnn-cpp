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
  public Hidden<ValueType, SizeAtCompileTime, 1, SizeAtCompileTime, 1>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, SizeAtCompileTime, 1, SizeAtCompileTime, 1>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /**
   * @brief Default constructor
   */
  Activation();
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

protected:
  FFNN_REGISTER_SERIALIZABLE(Activation)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

private:
  /// Layer activation units
  std::vector<NeuronType<ValueType>> neurons_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/activation.hpp>
#endif  // FFNN_LAYER_ACTIVATION_H
