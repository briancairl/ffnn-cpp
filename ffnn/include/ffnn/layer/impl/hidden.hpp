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
#define TARGS ValueType, LayerShape, _InputBlockType, _OutputBlockType

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
Hidden<TARGS>::Hidden(const ShapeType& input_shape, const ShapeType& output_shape) :
  Base(input_shape, output_shape),
  input_(NULL, input_shape.height, input_shape.width),
  output_(NULL,output_shape.height, output_shape.width),
  backward_error_(NULL, input_shape.height, input_shape.width),
  forward_error_(NULL, output_shape.height, output_shape.width)
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", input_shape);
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", output_shape);
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
Hidden<TARGS>::~Hidden()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", "Destroying [layer::Hidden] object <" << this->getID() << ">");
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
typename Hidden<TARGS>::OffsetType
Hidden<TARGS>::connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset)
{
  FFNN_ASSERT_MSG (Base::output_shape_ > 0, "Output dimensions are invalid (non-positive) or unresolved.");

  // Map output of next layer to input buffer
  auto output_ptr = const_cast<ValueType*>(next.getInputBuffer().data()) + offset;
  new (&output_) OutputMappingType(output_ptr, Base::output_shape_.height, Base::output_shape_.width);

  // Map error of next layer to backward-error buffer
  auto error_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data()) + offset;
  new (&forward_error_) OutputMappingType(error_ptr, Base::output_shape_.height, Base::output_shape_.width);

  // Return next offset after assigning buffer segments
  return offset + Base::outputShape().size();
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
bool Hidden<TARGS>::initialize()
{
  // Deduce input dimensions
  if (!Base::input_shape_.valid())
  {
    Base::input_shape_ = Base::evaluateInputSize();
  }

  FFNN_ASSERT_MSG (Base::input_shape_ > 0,  "Input dimensions are invalid (non-positive) or unresolved.");

  // Abort if layer is already initialized
  if (Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Do basic initialization
  if (Base::initialize())
  {
    // Create input buffer map
    auto input_ptr = const_cast<ValueType*>(Base::getInputBuffer().data());
    new (&input_) InputMappingType(input_ptr, Base::input_shape_.height, Base::input_shape_.width);

    // Create input buffer map
    auto error_ptr = const_cast<ValueType*>(Base::getBackwardErrorBuffer().data());
    new (&backward_error_) InputMappingType(error_ptr, Base::input_shape_.height, Base::input_shape_.width);

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == Base::inputShape().size())
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       Base::getID() <<
                       "> initialized as (in=" <<
                       Base::inputShape().size()  <<
                       ", out=" <<
                       Base::outputShape().size() <<
                       ")");
      return Base::isInitialized();
    }

    // Initialization failed
    Base::initialized_ = false;
    FFNN_ERROR_NAMED("layer::Hidden", "<" << Base::getID() << "> bad input count after input resolution.");
  }
  // Error initializing
  FFNN_ERROR_NAMED("layer::Hidden", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
void Hidden<TARGS>::save(typename Hidden<TARGS>::OutputArchive& ar,
                         typename Hidden<TARGS>::VersionType version) const
{
  ffnn::io::signature::apply<Hidden<TARGS>>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
void Hidden<TARGS>::load(typename Hidden<TARGS>::InputArchive& ar,
                         typename Hidden<TARGS>::VersionType version)
{
  ffnn::io::signature::check<Hidden<TARGS>>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
#undef TARGS
#endif  // FFNN_LAYER_IMPL_HIDDEN_HPP
