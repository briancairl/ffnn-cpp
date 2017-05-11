/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
{
#define HIDDEN_TARGS ValueType, LayerShape
#define HIDDEN_TARGS_ADVANCED _InputBlockType, _OutputBlockType
#define HIDDEN Hidden<HIDDEN_TARGS, HIDDEN_TARGS_ADVANCED>

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
HIDDEN::Hidden(const ShapeType& input_shape,
               const ShapeType& output_shape) :
  Base(input_shape, output_shape),
  input_(NULL, LayerShape::input_height, LayerShape::input_width),
  output_(NULL, LayerShape::output_height, LayerShape::output_width),
  backward_error_(NULL, LayerShape::input_height, LayerShape::input_width),
  forward_error_(NULL, LayerShape::output_height, LayerShape::output_width)
{
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
HIDDEN::~Hidden()
{
  FFNN_INTERNAL_DEBUG_NAMED("layer::Hidden", "Destroying [layer::Hidden] object <" << this->getID() << ">");
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
typename HIDDEN::OffsetType
HIDDEN::connectToForwardLayer(const Layer<ValueType>& next, OffsetType offset)
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
bool HIDDEN::initialize()
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
void HIDDEN::save(typename HIDDEN::OutputArchive& ar,
                  typename HIDDEN::VersionType version) const
{
  ffnn::io::signature::apply<HIDDEN>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template<typename ValueType,
         typename LayerShape,
         typename _InputBlockType,
         typename _OutputBlockType>
void HIDDEN::load(typename HIDDEN::InputArchive& ar,
                  typename HIDDEN::VersionType version)
{
  ffnn::io::signature::check<HIDDEN>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}

#undef HIDDEN_TARGS
#undef HIDDEN_TARGS_ADVANCED
#undef HIDDEN
}  // namespace layer
}  // namespace ffnn
