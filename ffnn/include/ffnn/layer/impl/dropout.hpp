/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// C++ Standard library
#include <ctime>
#include <cstring>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Dropout<ValueType, SizeAtCompileTime>::
Parameters::Parameters(const ValueType& probability,
                       bool blind) :
  probability(probability),
  blind(blind)
{
  FFNN_ASSERT_MSG(probability > 0, "[probability] must be greater than 0.");
  FFNN_ASSERT_MSG(probability < 1, "[probability] must be less than 1.");
}

template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Dropout<ValueType, SizeAtCompileTime>::
Dropout(const Parameters& config) :
  config_(config)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
Dropout<ValueType, SizeAtCompileTime>::~Dropout()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Dropout<ValueType, SizeAtCompileTime>::initialize()
{
  // This layer has equal inputs and outputs
  Base::output_dimension_ = Base::countInputs();

  // Do basic initialization
  if (Base::initialize())
  {    
    // Setup connectedness flags
    connected_.resize(Base::output_dimension_, true);

    FFNN_DEBUG_NAMED("layer::Dropout",
                     "<" <<
                     Base::getID() <<
                     "> initialized as (in=" <<
                     Base::input_dimension_ <<
                     ", out=" <<
                     Base::output_dimension_ <<
                     ")");

    return true;
  }
  return false;
}

template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Dropout<ValueType, SizeAtCompileTime>::forward()
{
  // Generate random values [-1, 1]
  typename Base::InputVector rand_vals;
  rand_vals.setRandom(Base::input_dimension_, 1);

  // Apply probabilistic dropout
  for (SizeType idx = 0; idx < Base::input_dimension_; idx++)
  {
    // // Generate random number and get probability of this number
    ProbabilityType p = (rand_vals(idx) + 1) < (2 * config_.probability);

    // // Flag connectedness
    connected_[idx] = (p < config_.probability);

    // Set output
    (*Base::output_)(idx) = connected_[idx] ? (*Base::input_)(idx) : 0;
  }
  return true;
}

template<typename ValueType,
         FFNN_SIZE_TYPE SizeAtCompileTime>
bool Dropout<ValueType, SizeAtCompileTime>::backward()
{
  if (config_.blind)
  {
    Base::backward_error_->noalias() = (*Base::forward_error_);
  }
  else
  {
    // Apply probabilistic dropout effects
    for (SizeType idx = 0; idx < Base::input_dimension_; idx++)
    {
      (*Base::backward_error_)(idx) = connected_[idx] ? (*Base::forward_error_)(idx) : 0;
    }
  }
  return true;
}
}  // namespace layer
}  // namespace ffnn
