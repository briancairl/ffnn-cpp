/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_LAYER_H
#define FFNN_LAYER_LAYER_H

// C++ Standard Library
#include <iostream>
#include <map>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/traits/shared.h>
#include <ffnn/traits/serializable.h>
#include <ffnn/traits/unique.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class Layer :
  public traits::Shared<Layer<ValueType>>,
  public traits::Unique
{
/// Connects Layer objects
template<typename LayerType>
friend bool connect(const typename LayerType::Ptr& from,
                    const typename LayerType::Ptr& to);
public:
  /// Scalar-type standardization
  typedef ValueType ScalarType;

  /// Size-type standardization
  typedef FFNN_SIZE_TYPE SizeType;

  /// Offset-type standardization
  typedef FFNN_OFFSET_TYPE OffsetType;

#ifndef FFNN_NO_EXPLICIT_ALIGNMENT
  /// Data buffer (vector) type
  typedef std::vector<ValueType, Eigen::aligned_allocator<ValueType>> BufferType;
#else
  /// Data buffer (vector) type
  typedef std::vector<ValueType> BufferType;
#endif

  /**
   * @brief Setup constructor
   * @param input_dim  number of inputs to the Layer
   * @param output_dim  number of outputs from the Layer
   */
  Layer(SizeType input_dim = 0, SizeType output_dim = 0);
  virtual ~Layer();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

  /**
   * @brief Returns true if layer has been initialized
   * @retval true  if layer is initialized
   * @retval false  otherwise
   */
  virtual bool isInitialized();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward()
  {
    return true;
  }

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool backward()
  {
    return true;
  }

  /**
   * @brief Applies layer weight updates
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   */
  virtual bool update()
  {
    return true;
  }

  /**
   * @brief Exposes raw input buffer
   */
  inline const BufferType& getInputBuffer() const
  {
    return input_buffer_;
  }

  /**
   * @brief Exposes raw bakcward-error buffer
   */
  inline const BufferType& getBackwardErrorBuffer() const
  {
    return backward_error_buffer_;
  }

  /**
   * @brief Returns the total number of Layer inputs
   */
  inline SizeType getInputDim() const
  {
    return input_dimension_;
  }

  /**
   * @brief Returns the total number of Layer outputs
   */
  inline SizeType getOutputDim() const
  {
    return output_dimension_;
  }

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_dimension_</code>
   */
  virtual OffsetType connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset) = 0;

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
  SizeType countInputs() const;

  /**
   * @brief Connects all previous layer to Layer input
   * @retval <code>input_dimension_</code>
   */
  OffsetType connectInputLayers();

  /// Flags if a layer has been initialized
  bool initialized_;

  /// Flags if a layer was loaded from file
  bool loaded_;

  /// Pointers to previous layers
  std::map<std::string, typename Layer<ValueType>::Ptr> prev_;

  /// Total number of input connections
  SizeType input_dimension_;

  /// Total number of output connections
  SizeType output_dimension_;

  /// Raw input value buffer
  BufferType input_buffer_;

  /// Raw bakward error value buffer
  BufferType backward_error_buffer_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/layer.hpp>
#endif  // FFNN_LAYER_LAYER_H
