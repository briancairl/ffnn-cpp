/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_INTERFACE_H
#define FFNN_LAYER_INTERNAL_INTERFACE_H

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>
#include <ffnn/internal/traits/unique.h>
#include <ffnn/internal/signature.h>
#include <ffnn/layer/internal/shape.h>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace layer
{
namespace internal
{
/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class Interface :
  public traits::Unique
{
public:
  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef FFNN_SIZE_TYPE SizeType;

  /// Offset type standardization
  typedef FFNN_OFFSET_TYPE OffsetType;

  /// Dimension type standardization
  typedef Shape<SizeType> ShapeType;

  explicit
  Interface(const ShapeType& input_shape  = ShapeType(Eigen::Dynamic),
            const ShapeType& output_shape = ShapeType(Eigen::Dynamic)) :
    initialized_(false),
    setup_required_(true),
    input_shape_(input_shape),
    output_shape_(output_shape)
  {}
  virtual ~Interface()
  {
    FFNN_INTERNAL_DEBUG_NAMED("layer::Interface", "Destroying [layer::Interface] object <" << this->getID() << ">");
  }

  /**
   * @brief Returns the total number of Interface inputs
   */
  inline SizeType inputSize() const
  {
    return input_shape_.size();
  }

  /**
   * @brief Returns the total number of Interface outputs
   */
  inline SizeType outputSize() const
  {
    return output_shape_.size();
  }

  /**
   * @brief Returns the Interface input dimension oject
   */
  inline const ShapeType& inputShape() const
  {
    return input_shape_;
  }

  /**
   * @brief Returns the Interface output dimension oject
   */
  inline const ShapeType& outputShape() const
  {
    return output_shape_;
  }

  /**
   * @brief Returns the total number counted (evaluated) inputs
   */
  virtual SizeType evaluateInputSize() const
  {
    return input_shape_.size();
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
  FFNN_REGISTER_SERIALIZABLE(Interface)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const
  {
    ffnn::io::signature::apply<Interface<ValueType>>(ar);
    traits::Unique::save(ar, version);

    // Save flags
    ar & initialized_;

    // Save dimensions
    ar & input_shape_;
    ar & output_shape_;
  }

  /// Load serializer
  void load(InputArchive& ar, VersionType version)
  {
    ffnn::io::signature::check<Interface<ValueType>>(ar);
    traits::Unique::load(ar, version);

    // Load flags
    ar & initialized_;

    // Load dimensions
    ar & input_shape_;
    ar & output_shape_;

    setup_required_ = false;
  }

  /// Flags which indicated that layer has been initialized
  bool initialized_;

  /// Flags which indicates that layer should be initialized normally
  bool setup_required_;

  /// Total number of input connections
  ShapeType input_shape_;

  /// Total number of output connections
  ShapeType output_shape_;
};
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_INTERFACE_H
