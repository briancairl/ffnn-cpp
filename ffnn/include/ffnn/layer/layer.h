/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_LAYER_H
#define FFNN_LAYER_LAYER_H

// C++ Standard Library
#include <map>

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>

// FFNN
#include <ffnn/aligned_types.h>
#include <ffnn/config/global.h>
#include <ffnn/layer/internal/layer_base.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class Layer :
  public internal::LayerBase<ValueType>
{
/// Connects Layer objects
template<typename LayerType>
friend bool connect(const typename LayerType::Ptr& from,
                    const typename LayerType::Ptr& to);
public:
  /// Base type alias
  using Base = internal::LayerBase<ValueType>;

  /// Shared resource standardization
  typedef boost::shared_ptr<Layer> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const Layer> ConstPtr;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::DimType DimType;

  /**
   * @brief Setup constructor
   * @param input_size  number of inputs to the Layer
   * @param output_size  number of outputs from the Layer
   */
  Layer(DimType input_dim = 0, DimType output_dim = 0);
  virtual ~Layer();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

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

  inline const aligned::Buffer<ValueType>& getInputBuffer() const
  {
    return input_buffer_;
  }

  inline const aligned::Buffer<ValueType>& getBackwardErrorBuffer() const
  {
    return backward_error_buffer_;
  }

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_dim_.size()</code>
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
  SizeType evaluateInputSize() const;

  /**
   * @brief Connects all previous layer to Layer input
   * @retval <code>input_dim_.size()</code>
   */
  OffsetType connectInputLayers();

  /// Pointers to previous layers
  std::map<std::string, typename Layer<ValueType>::Ptr> prev_;

  /// Raw input value buffer
  aligned::Buffer<ValueType> input_buffer_;

  /// Raw bakward error value buffer
  aligned::Buffer<ValueType> backward_error_buffer_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/layer.hpp>
#endif  // FFNN_LAYER_LAYER_H
