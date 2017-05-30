/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_LAYER_H
#define FFNN_LAYER_LAYER_H

// C++ Standard Library
#include <map>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/internal/serializable.h>
#include <ffnn/internal/unique.h>
#include <ffnn/layer/internal/shape.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class Layer :
  public internal::Unique
{
/// Connects Layer objects
template<typename LayerType>
friend bool connect(const typename LayerType::Ptr& from,
                    const typename LayerType::Ptr& to);
public:
  /// Base type alias
  using Base = internal::Unique;

  /// Self type alias
  using Self = Layer<ValueType>;

  /// Shared resource standardization
  typedef boost::shared_ptr<Self> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const Self> ConstPtr;

  /// Buffer type standardization
  typedef std::vector<ValueType, Eigen::aligned_allocator<ValueType>> BufferType;

  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef FFNN_SIZE_TYPE SizeType;

  /// Offset type standardization
  typedef FFNN_OFFSET_TYPE OffsetType;

  /// Dimension type standardization
  typedef Shape<SizeType> ShapeType;

  /**
   * @brief Setup constructor
   * @param input_shape  number of inputs to the Layer
   * @param output_shape  number of outputs from the Layer
   */
  Layer(const ShapeType& input_shape, const ShapeType& output_shape);
  virtual ~Layer();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  virtual bool update() = 0;

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward() = 0;

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool backward() = 0;

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_shape_.size()</code>
   */
  virtual OffsetType connectToForwardLayer(const Self& next, OffsetType offset) = 0;

  /**
   * @brief Exposes const reference to raw data input buffer
   */
  inline const BufferType& getInputBuffer() const
  {
    return input_buffer_;
  }

  /**
   * @brief Exposes const reference to raw data backward-error buffer
   */
  inline const BufferType& getBackwardErrorBuffer() const
  {
    return backward_error_buffer_;
  }

  /**
   * @brief Returns the Interface input dimension oject
   */
  inline const ShapeType& getInputShape() const
  {
    return input_shape_;
  }

  /**
   * @brief Returns the Interface output dimension oject
   */
  inline const ShapeType& getOutputShape() const
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
  // Register serialization material
  FFNN_REGISTER_SERIALIZABLE(Layer)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

  /**
   * @brief Counts the number of inputs from outputs of previous layers
   * @return total input count
   */
  SizeType evaluateInputSize() const;

  /**
   * @brief Connects all previous layer to Layer input
   * @retval <code>input_shape_.size()</code>
   */
  OffsetType connectInputLayers();

  /// Pointers to previous layers
  std::map<std::string, typename Self::Ptr> prev_;

  /// Raw input value buffer
  BufferType input_buffer_;

  /// Raw bakward error value buffer
  BufferType backward_error_buffer_;

  /// Total number of input connections
  ShapeType input_shape_;

  /// Total number of output connections
  ShapeType output_shape_;

  /// Flags which indicated that layer has been initialized
  bool initialized_;

  /// Flags which indicates that layer should be initialized normally
  bool setup_required_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/layer.hpp>
#endif  // FFNN_LAYER_LAYER_H
