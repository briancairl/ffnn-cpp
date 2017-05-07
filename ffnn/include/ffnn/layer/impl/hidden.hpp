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
#define HIDDEN_TARGS ValueType, InputsHeightAtCompileTime, InputsWidthAtCompileTime, OutputsHeightAtCompileTime, OutputsWidthAtCompileTime
#define HIDDEN_TARGS_ADVANCED _InputBlockType, _OutputBlockType, _InputMappingType, _OutputMappingType
#define HIDDEN Hidden<HIDDEN_TARGS, HIDDEN_TARGS_ADVANCED>

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::Hidden(const ShapeType& input_shape,
               const ShapeType& output_shape) :
  Base(input_shape, output_shape)
{
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::~Hidden()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
typename HIDDEN::OffsetType 
HIDDEN::connectToForwardLayer(const Base& next, OffsetType offset)
{
  FFNN_ASSERT_MSG (Base::output_shape_ > 0, "Output dimensions are invalid (non-positive) or unresolved.");

  // Map output of next layer to input buffer
  auto output_ptr = const_cast<ValueType*>(next.getInputBuffer().data()) + offset;
  output_ = _OutputMappingType::create(output_ptr,
                                       Base::output_shape_.height,
                                       Base::output_shape_.width);

  // Map error of next layer to backward-error buffer
  auto error_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data()) + offset;
  forward_error_ = _OutputMappingType::create(error_ptr,
                                              Base::output_shape_.height,
                                              Base::output_shape_.width);

  // Return next offset after assigning buffer segments
  return offset + Base::outputSize();
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
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
    input_ = _InputMappingType::create(input_ptr,
                                       Base::input_shape_.height,
                                       Base::input_shape_.width);

    // Create input buffer map
    auto error_ptr = const_cast<ValueType*>(Base::getBackwardErrorBuffer().data());
    backward_error_ = _InputMappingType::create(error_ptr,
                                                Base::input_shape_.height,
                                                Base::input_shape_.width);

    FFNN_DEBUG_NAMED("layer::Hidden", "Created forward mappings.");

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == Base::inputSize())
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       Base::getID() <<
                       "> initialized as (in=" <<
                       Base::inputSize()  <<
                       ", out=" <<
                       Base::outputSize() <<
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
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
void HIDDEN::save(typename HIDDEN::OutputArchive& ar,
                  typename HIDDEN::VersionType version) const
{
  ffnn::io::signature::apply<HIDDEN>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
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
