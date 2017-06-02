/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
#ifndef FFNN_LAYER_IMPL_HIDDEN_HPP
#define FFNN_LAYER_IMPL_HIDDEN_HPP

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
template <typename ValueType,
          typename Options,
          typename Extrinsics>
Hidden<ValueType, Options, Extrinsics>::Hidden(const ShapeType& input_shape, const ShapeType& output_shape) :
  BaseType(input_shape, output_shape),
  input_(NULL, input_shape.height, input_shape.width),
  output_(NULL,output_shape.height, output_shape.width),
  backward_error_(NULL, input_shape.height, input_shape.width),
  forward_error_(NULL, output_shape.height, output_shape.width)
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", "[" << input_shape << " | " << output_shape << "]");
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
Hidden<ValueType, Options, Extrinsics>::~Hidden()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", "Destroying [layer::Hidden] object <" << this->getID() << ">");
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
offset_type Hidden<ValueType, Options, Extrinsics>::connectToForwardLayer(const Layer<ValueType>& next, offset_type offset)
{
  FFNN_ASSERT_MSG (BaseType::output_shape_ > 0, "Output dimensions are invalid (non-positive) or unresolved.");

  // Map output of next layer to input buffer
  auto output_ptr = const_cast<ValueType*>(next.getInputBuffer().data()) + offset;
  new (&output_) OutputMappingType(output_ptr, BaseType::output_shape_.height, BaseType::output_shape_.width);

  // Map error of next layer to backward-error buffer
  auto error_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data()) + offset;
  new (&forward_error_) OutputMappingType(error_ptr, BaseType::output_shape_.height, BaseType::output_shape_.width);

  // Return next offset after assigning buffer segments
  return offset + BaseType::getOutputShape().size();
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
bool Hidden<ValueType, Options, Extrinsics>::initialize()
{
  // Deduce input dimensions
  if (!BaseType::input_shape_.valid())
  {
    BaseType::input_shape_ = BaseType::evaluateInputSize();
  }

  FFNN_ASSERT_MSG (BaseType::input_shape_ > 0,  "Input dimensions are invalid (non-positive) or unresolved.");

  // Abort if layer is already initialized
  if (BaseType::setupRequired() && BaseType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << BaseType::getID() << "> already initialized.");
    return false;
  }

  // Do basic initialization
  if (BaseType::initialize())
  {
    // Create input buffer map
    auto input_ptr = const_cast<ValueType*>(BaseType::getInputBuffer().data());
    new (&input_) InputMappingType(input_ptr, BaseType::input_shape_.height, BaseType::input_shape_.width);

    // Create input buffer map
    auto error_ptr = const_cast<ValueType*>(BaseType::getBackwardErrorBuffer().data());
    new (&backward_error_) InputMappingType(error_ptr, BaseType::input_shape_.height, BaseType::input_shape_.width);

    // Resolve previous layer output buffers
    if (BaseType::connectInputLayers() == BaseType::getInputShape().size())
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       BaseType::getID() <<
                       "> initialized as (in=" <<
                       BaseType::getInputShape().size()  <<
                       ", out=" <<
                       BaseType::getOutputShape().size() <<
                       ")");
      return BaseType::isInitialized();
    }

    // Initialization failed
    BaseType::initialized_ = false;
    FFNN_ERROR_NAMED("layer::Hidden", "<" << BaseType::getID() << "> bad input count after input resolution.");
  }
  // Error initializing
  FFNN_ERROR_NAMED("layer::Hidden", "<" << BaseType::getID() << "> failed to initialize.");
  return false;
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void Hidden<ValueType, Options, Extrinsics>::save(typename Hidden<ValueType, Options, Extrinsics>::OutputArchive& ar,
                                                  typename Hidden<ValueType, Options, Extrinsics>::VersionType version) const
{
  ffnn::internal::signature::apply<SelfType>(ar);
  BaseType::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template <typename ValueType,
          typename Options,
          typename Extrinsics>
void Hidden<ValueType, Options, Extrinsics>::load(typename Hidden<ValueType, Options, Extrinsics>::InputArchive& ar,
                                                  typename Hidden<ValueType, Options, Extrinsics>::VersionType version)
{
  ffnn::internal::signature::check<SelfType>(ar);
  BaseType::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_IMPL_HIDDEN_HPP
