/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_HIDDEN_H
#define FFNN_LAYER_HIDDEN_H

// C++ Standard Library
#include <vector>

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/layer/layer.h>
#include <ffnn/aligned_types.h>


namespace ffnn
{
namespace layer
{
/**
 * @brief A network hidden-layer object
 */
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE OutputsAtCompileTime = Eigen::Dynamic,
         typename _InputVectorType = Eigen::Matrix<ValueType, InputsAtCompileTime, 1, Eigen::ColMajor>,
         typename _OutputVectorType = Eigen::Matrix<ValueType, OutputsAtCompileTime, 1, Eigen::ColMajor>,
         typename _InputMappingType = aligned::Map<_InputVectorType>,
         typename _OutputMappingType = aligned::Map<_OutputVectorType>>
class Hidden :
  public Layer<ValueType>
{
public:
  /// Base type alias
  using Base = Layer<ValueType>;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Hidden input type standardization
  typedef _InputVectorType InputVectorType;

  /// Hidden output type standardization
  typedef _OutputVectorType OutputVectorType;

  /**
   * @brief Setup constructor
   * @param input_dim  number of inputs to the Hidden
   * @param output_dim  number of outputs from the Hidden
   */
  explicit
  Hidden(SizeType input_dim = InputsAtCompileTime,
         SizeType output_dim = OutputsAtCompileTime);
  virtual ~Hidden();

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

protected:
  FFNN_REGISTER_SERIALIZABLE(Layer)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);

  /// Memory-mapped input vector
  typename _InputMappingType::Ptr input_;

  /// Memory-mapped output vector
  typename _OutputMappingType::Ptr output_;

  /// Backward error vector
  typename _InputMappingType::Ptr backward_error_;

  /// Output-target error vector
  typename _OutputMappingType::Ptr forward_error_;

private:
  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_dimension_</code>
   */
  OffsetType connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset);
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/hidden.hpp>
#endif  // FFNN_LAYER_HIDDEN_H
