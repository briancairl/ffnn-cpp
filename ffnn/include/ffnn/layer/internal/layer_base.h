/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H
#define FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>
#include <ffnn/internal/traits/unique.h>
#include <ffnn/internal/signature.h>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class LayerBase :
  public traits::Unique
{
public:
  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef FFNN_SIZE_TYPE SizeType;

  /// Offset type standardization
  typedef FFNN_OFFSET_TYPE OffsetType;

  LayerBase(SizeType input_size, SizeType output_size) :
    initialized_(false),
    setup_required_(false),
    input_size_(input_size > 0 ? input_size : 0),
    output_size_(output_size > 0 ? output_size : 0)
  {}
  virtual ~LayerBase() {}

  /**
   * @brief Returns the total number of Layer inputs
   */
  SizeType inputSize() const
  {
    return input_size_;
  }

  /**
   * @brief Returns the total number of Layer outputs
   */
  SizeType outputSize() const
  {
    return output_size_;
  }

  /**
   * @brief Returns the total number counted (evaluated) inputs
   */
  virtual SizeType evaluateInputSize() const
  {
    return input_size_;
  }

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize() = 0;

  /**
   * @brief Returns true if layer has been initialized
   * @retval true  if layer is initialized
   * @retval false  otherwise
   */
  virtual bool isInitialized() const
  {
    return initialized_;
  }

  /**
   * @brief Returns true if portions of the interface must be setup
   * @retval true  if reinitialization allowed
   * @retval false  otherwise
   */
  bool setupRequired() const
  {
    return setup_required_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(LayerBase)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const
  {
    ffnn::io::signature::apply<LayerBase<ValueType>>(ar);
    traits::Unique::save(ar, version);
  }

  /// Load serializer
  void load(InputArchive& ar, VersionType version)
  {
    ffnn::io::signature::check<LayerBase<ValueType>>(ar);
    traits::Unique::load(ar, version);
  }

  /// Flags which indicated that layer has been initialized
  bool initialized_;

  /// Flags which indicates that layer should be initialized normally
  bool setup_required_;

  /// Total number of input connections
  SizeType input_size_;

  /// Total number of output connections
  SizeType output_size_;
};
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H
