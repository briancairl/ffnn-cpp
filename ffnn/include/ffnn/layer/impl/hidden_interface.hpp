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
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  HiddenInterface(SizeType input_size, SizeType output_size) :
    _ForwardInterfacedType(input_size, output_size)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  ~HiddenInterface()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
typename HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  OffsetType 
HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  connectToForwardLayer(const _ForwardInterfacedType& next, OffsetType offset)
{
  // Map output of next layer to input buffer
  output_ = _OutputMappingType::create(next.getInputPtr() + offset, _ForwardInterfacedType::outputSize());      

  // Map error of next layer to backward-error buffer
  forward_error_ = _OutputMappingType::create(next.getBackwardErrorPtr() + offset, _ForwardInterfacedType::outputSize());

  // Return next offset after assigning buffer segments
  return offset + _ForwardInterfacedType::outputSize();
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
bool HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  initialize()
{
  // Abort if layer is already initialized
  if (!_ForwardInterfacedType::setupRequired() &&
      _ForwardInterfacedType::isInitialized())
  {
    FFNN_WARN_NAMED("layer::HiddenInterface", "<" << _ForwardInterfacedType::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  _ForwardInterfacedType::input_size_ = _ForwardInterfacedType::evaluateInputSize();

  // Validate and set input count
  FFNN_STATIC_ASSERT_MSG (InputsAtCompileTime < 0 ||
                          InputsAtCompileTime == _ForwardInterfacedType::input_size_,
                          "(InputsAtCompileTime != `resolved input size`) for fixed-size layer.");
    
  // Do basic initialization
  if (_ForwardInterfacedType::initialize())
  {
    FFNN_DEBUG_NAMED("layer::HiddenInterface", "Creating forward mappings.");

    // Create input buffer map
    input_ = _InputMappingType::create(_ForwardInterfacedType::getInputPtr(),
                                       _ForwardInterfacedType::input_size_);

    // Create input buffer map
    backward_error_ = _InputMappingType::create(_ForwardInterfacedType::getBackwardErrorPtr(),
                                                _ForwardInterfacedType::input_size_);

    // Resolve previous layer output buffers
    if (_ForwardInterfacedType::connectInputLayers() == _ForwardInterfacedType::input_size_)
    {
      FFNN_DEBUG_NAMED("layer::HiddenInterface",
                       "<" <<
                       _ForwardInterfacedType::getID() <<
                       "> initialized as (in=" <<
                       _ForwardInterfacedType::input_size_  <<
                       ", out=" <<
                       _ForwardInterfacedType::outputSize() <<
                       ")");
      return _ForwardInterfacedType::isInitialized();
    }

    // Initialization failed
    _ForwardInterfacedType::initialized_ = false;
    FFNN_ERROR_NAMED("layer::HiddenInterface", "<" << _ForwardInterfacedType::getID() << "> bad input count after input resolution.");
  }
  // Error initializing
  FFNN_ERROR_NAMED("layer::HiddenInterface", "<" << _ForwardInterfacedType::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
void HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  save(typename HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::OutputArchive& ar,
       typename HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::VersionType version) const
{
  ffnn::io::signature::apply<HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>>(ar);
  _ForwardInterfacedType::save(ar, version);
  FFNN_DEBUG_NAMED("layer::HiddenInterface", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType,
         typename _ForwardInterfacedType>
void HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::
  load(typename HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::InputArchive& ar,
       typename HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>::VersionType version)
{
  ffnn::io::signature::check<HiddenInterface<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType, _ForwardInterfacedType>>(ar);
  _ForwardInterfacedType::load(ar, version);
  FFNN_DEBUG_NAMED("layer::HiddenInterface", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
