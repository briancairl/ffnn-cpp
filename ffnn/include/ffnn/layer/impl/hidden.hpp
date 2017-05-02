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
#define HIDDEN Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime, _InputVectorType, _OutputVectorType, _InputMappingType, _OutputMappingType>

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::Hidden(SizeType input_size, SizeType output_size) :
    Base(input_size, output_size)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::~Hidden()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
typename HIDDEN::OffsetType 
HIDDEN::connectToForwardLayer(const Base& next, OffsetType offset)
{
  // Map output of next layer to input buffer
  output_ = _OutputMappingType::create(next.getInputPtr() + offset, Base::outputSize());      

  // Map error of next layer to backward-error buffer
  forward_error_ = _OutputMappingType::create(next.getBackwardErrorPtr() + offset, Base::outputSize());

  // Return next offset after assigning buffer segments
  return offset + Base::outputSize();
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
bool HIDDEN::initialize()
{
  // Abort if layer is already initialized
  if (!Base::setupRequired() &&
      Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  Base::input_size_ = Base::evaluateInputSize();

  // Validate and set input count
  FFNN_STATIC_ASSERT_MSG (InputsAtCompileTime < 0 ||
                          InputsAtCompileTime == Base::input_size_,
                          "(InputsAtCompileTime != `resolved input size`) for fixed-size layer.");
    
  // Do basic initialization
  if (Base::initialize())
  {
    FFNN_DEBUG_NAMED("layer::Hidden", "Creating forward mappings.");

    // Create input buffer map
    input_ = _InputMappingType::create(Base::getInputPtr(),
                                       Base::input_size_);

    // Create input buffer map
    backward_error_ = _InputMappingType::create(Base::getBackwardErrorPtr(),
                                                Base::input_size_);

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == Base::input_size_)
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       Base::getID() <<
                       "> initialized as (in=" <<
                       Base::input_size_  <<
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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
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
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime,
         typename _InputVectorType,
         typename _OutputVectorType,
         typename _InputMappingType,
         typename _OutputMappingType>
void HIDDEN::load(typename HIDDEN::InputArchive& ar,
                            typename HIDDEN::VersionType version)
{
  ffnn::io::signature::check<HIDDEN>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}

#undef HIDDEN
}  // namespace layer
}  // namespace ffnn
