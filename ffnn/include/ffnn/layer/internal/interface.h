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
#include <ffnn/layer/internal/dimensions.h>

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
  typedef Dimensions<SizeType> DimType;

  explicit
  Interface(const DimType& input_dim  = DimType(Eigen::Dynamic),
            const DimType& output_dim = DimType(Eigen::Dynamic)) :
    initialized_(false),
    setup_required_(true),
    input_dim_(input_dim),
    output_dim_(output_dim)
  {}
  virtual ~Interface() {}

  /**
   * @brief Returns the total number of Interface inputs
   */
  inline SizeType inputSize() const
  {
    return input_dim_.size();
  }

  /**
   * @brief Returns the total number of Interface outputs
   */
  inline SizeType outputSize() const
  {
    return output_dim_.size();
  }

  /**
   * @brief Returns the Interface input dimension oject
   */
  inline const DimType& inputDim() const
  {
    return input_dim_;
  }

  /**
   * @brief Returns the Interface output dimension oject
   */
  inline const DimType& outputDim() const
  {
    return output_dim_;
  }


  /**
   * @brief Returns the total number counted (evaluated) inputs
   */
  virtual SizeType evaluateInputSize() const
  {
    return input_dim_.size();
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
    input_dim_.save(ar, version);
    output_dim_.save(ar, version);
  }

  /// Load serializer
  void load(InputArchive& ar, VersionType version)
  {
    ffnn::io::signature::check<Interface<ValueType>>(ar);
    traits::Unique::load(ar, version);

    // Load flags
    ar & initialized_;

    // Load dimensions
    input_dim_.load(ar, version);
    output_dim_.load(ar, version);

    setup_required_ = false;
  }

  /// Flags which indicated that layer has been initialized
  bool initialized_;

  /// Flags which indicates that layer should be initialized normally
  bool setup_required_;

  /// Total number of input connections
  DimType input_dim_;

  /// Total number of output connections
  DimType output_dim_;
};
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_INTERFACE_H
