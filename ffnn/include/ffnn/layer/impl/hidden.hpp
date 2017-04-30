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
         typename _OutputMappingType>
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  Hidden(SizeType input_dim, SizeType output_dim) :
    Base(input_dim, output_dim)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  ~Hidden()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  OffsetType 
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  connectToForwardLayer(const Base& next, OffsetType offset)
{
  // Map output of next layer to input buffer
  {
    ValueType* ptr = const_cast<ValueType*>(next.getInputBuffer().data());
    output_ = _OutputMappingType::create(ptr + offset, Base::output_dimension_);      
  }
  // Map error of next layer to backward-error buffer
  {
    ValueType* ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
    forward_error_ = _OutputMappingType::create(ptr + offset, Base::output_dimension_);
  }
  // Return next offset after assigning buffer segments
  return offset + Base::output_dimension_;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
bool Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  initialize()
{
  // Abort if layer is already initialized
  if (!Base::loaded_ && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  Base::input_dimension_ = Base::countInputs();

  // Validate and set input count
  FFNN_STATIC_ASSERT_MSG (InputsAtCompileTime < 0 ||
                          InputsAtCompileTime == Base::input_dimension_,
                          "(InputsAtCompileTime != `resolved input size`) for fixed-size layer.");
    
  // Do basic initialization
  if (Base::initialize())
  {
    FFNN_DEBUG_NAMED("layer::Hidden", "Creating forward mappings.");

    // Create input buffer map
    input_ = _InputMappingType::create(Base::input_buffer_.data(), Base::input_dimension_);

    // Create input buffer map
    backward_error_ = _InputMappingType::create(Base::backward_error_buffer_.data(), Base::input_dimension_);

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == Base::input_dimension_)
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       Base::getID() <<
                       "> initialized as (in=" <<
                       Base::input_dimension_  <<
                       ", out=" <<
                       Base::output_dimension_ <<
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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
void Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  save(typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::OutputArchive& ar,
       typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::VersionType version) const
{
  ffnn::io::signature::apply<Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
void Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::
  load(typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::InputArchive& ar,
       typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>::VersionType version)
{
  ffnn::io::signature::check<Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}
}  // namespace layer
}  // namespace ffnn
