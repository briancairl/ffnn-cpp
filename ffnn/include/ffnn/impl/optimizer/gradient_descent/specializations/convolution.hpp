/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_CONVOLUTION_HPP
#define FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_CONVOLUTION_HPP

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/layer/convolution.h>
#include <ffnn/layer/convolution/sizing.h>

namespace ffnn
{
namespace optimizer
{
using layer::convolution::EmbeddingMode;
using layer::convolution::ColEmbedding;
using layer::convolution::RowEmbedding;

template<>
template <typename ValueType,
          typename Options,
          typename Extrinsics>
class GradientDescent<layer::Convolution<ValueType, Options, Extrinsics>, CrossEntropy> :
  public GradientDescent_<layer::Convolution<ValueType, Options, Extrinsics>, CrossEntropy>
{
public:
  /// Layer type alias
  using LayerType = layer::Convolution<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType  = GradientDescent_<LayerType, CrossEntropy>;

  /// Layer shape type alias
  using ShapeType = typename LayerType::ShapeType;

  // Use BaseType constructors
  using BaseType::BaseType;

  /**
   * @brief Computes one gradient update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimizer setp was successful
   * @retval false  otherwise
   */
  bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Get block dimensions
    const ShapeType& _is = layer.config_.embedded_input_shape_;
    const ShapeType& _os = layer.config_.output_shape_;
    const ShapeType& _ss = layer.config_.stride_shape_;

    // Reset weight delta
    for (offset_type idx = 0, hdx = 0; idx < _os.height; idx++, hdx += _ss.height)
    {
      for (offset_type jdx = 0, wdx = 0; jdx < _os.width; jdx++, wdx += _ss.width)
      {
        for (offset_type kdx = 0; kdx < static_cast<offset_type>(BaseType::gradient_.size()); kdx++)
        {
          // Remap indices
          offset_type u_idx, u_jdx;
          remap<Options::embedding_mode>(_os, idx, jdx, kdx, u_idx, u_jdx);

          // Accumulate kernel gradient updates
          BaseType::gradient_[kdx] +=
            BaseType::prev_input_.block(hdx, wdx, _is.height, _is.width) *
            layer.forward_error_(u_idx, u_jdx);

          // Accumulate bias gradient updates
          BaseType::gradient_.bias += layer.forward_error_(u_idx, u_jdx);
        }
      }
    }

    // Back-prop error
    return true;
  }

private:
  /// Helper to remap filter/input indices
  template<EmbeddingMode T>
  typename std::enable_if<T == ColEmbedding, void>::type
    remap(const ShapeType& s, offset_type idx, offset_type jdx, offset_type kdx, offset_type& u_idx, offset_type& u_jdx)
  {
    u_idx = idx * s.depth + kdx;
    u_jdx = jdx;
  }
  template<EmbeddingMode T>
  typename std::enable_if<T == RowEmbedding, void>::type
    remap(const ShapeType& s, offset_type idx, offset_type jdx, offset_type kdx, offset_type& u_idx, offset_type& u_jdx)
  {
    u_idx = idx;
    u_jdx = jdx * s.depth + kdx;
  }
};
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_IMPL_SPECIALIZATIONS_LAYER_GRADIENT_DESCENT_CONVOLUTION_HPP