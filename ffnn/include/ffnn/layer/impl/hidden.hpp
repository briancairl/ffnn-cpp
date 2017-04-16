/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::Hidden(SizeType input_dim, SizeType output_dim) :
  Base(input_dim, output_dim)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::~Hidden()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::setup()
{
  FFNN_DEBUG_NAMED("layer::Hidden",
                   "<"  <<
                   Base::id() <<
                   "> initialized as (in=" <<
                   Base::input_dimension_  <<
                   ", out=" <<
                   Base::output_dimension_ << ")");
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
typename Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::OffsetType 
Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  // Map output of next layer to input buffer
  {
    ValueType* ptr = const_cast<ValueType*>(next.getInputBuffer().data());
    output_ = Mapped<OutputVector>::create(ptr + offset, Base::output_dimension_);      
  }
  // Map error of next layer to backward-error buffer
  {
    ValueType* ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data());
    forward_error_ = Mapped<OutputVector>::create(ptr + offset, Base::output_dimension_);
  }
  // Return next offset after assigning buffer segments
  return offset + Base::output_dimension_;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool Hidden<ValueType, InputsAtCompileTime, OutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << Base::id() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  SizeType input_count = Base::countInputs();
  {
    // Validate and set input count
    FFNN_ASSERT_MSG (InputsAtCompileTime < 0 || input_count == InputsAtCompileTime,
                     "(InputsAtCompileTime != `resolved input size`) for fixed-size layer.");
    
    // Set input count (including bias unit)
    Base::input_dimension_ = input_count + 1;
  }

  // Do basic initialization
  if (Base::initialize())
  {
    // Create input buffer map
    input_ = Mapped<InputVector>::create(Base::input_buffer_);

    // Create input buffer map
    backward_error_ = Mapped<InputVector>::create(Base::backward_error_buffer_);

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == input_count)
    {
      // Run Hidden setup
      return setup() && isInitialized();
    }
  }

  // Error initializing
  Base::initialized_ = false;
  FFNN_ERROR_NAMED("layer::Hidden", "< " << Base::id() << "> failed basic initialization.");
  return false;
}
}  // namespace layer
}  // namespace ffnn
