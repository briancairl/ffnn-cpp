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
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::
FullyConnected(SizeType output_dim, const Config& config) :
  Base(1 /*Bias Unit*/, output_dim),
  config_(config)
{
  FFNN_ASSERT_MSG(config.learning_rate > 0, "'config.learning_rate' is non-positive");
  FFNN_ASSERT_MSG(config.weight_init_variance > 0, "'config.weight_init_variance' is non-positive");
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::~FullyConnected()
{}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::forward()
{
  // Copy current input for updateing
  prev_input_ = Base::input();

  // Compute weighted outputs
  w_input_.noalias() = w_ * prev_input_;

  // Compute neuron outputs
  for (SizeType idx = 0; idx < Base::output_dimension_; idx++)
  {
    neurons_[idx]->fn(w_input_(idx), (*Base::output_)(idx));
  }
  return true;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::backward()
{
  // Compute neuron derivatives
  OutputVector deriv = Base::output();
  for (SizeType idx = 0; idx < Base::output_dimension_; idx++)
  {
    // Compute activation derivate
    neurons_[idx]->derivative(w_input_(idx), deriv(idx));
  }

  // Incorporate error
  deriv.array() *= Base::forward_error_->array();

  // Compute current weight delta
  WeightMatrix w_delta_curr(deriv.rows(), prev_input_.rows());
  w_delta_curr.noalias() = deriv * prev_input_.transpose();

  // Accumulate weight delta
  w_delta_ += w_delta_curr;

  // Compute back-propagated error
  (*Base::backward_error_) = w_.transpose() * w_delta_curr;
  return true;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::update()
{
  // Update weights
  w_.noalias() -= config_.learning_rate * w_delta_;

  // Reset weight delta
  w_delta_.setZero(Base::output_dimension_, Base::input_dimension_);
  return true;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
template<class Archive>
void FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::
  save(Archive& ar, ClassVersionType version) const
{
  // Save config parameters
  ar & config_.learning_rate;
  ar & config_.weight_init_variance;

  // Save weight matrix
  ar & w_;
  ar & w_delta_;
  ar & w_input_;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
template<class Archive>
void FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::
  load(Archive& ar, ClassVersionType version)
{
  // Load config parameters
  ar & config_.learning_rate;
  ar & config_.weight_init_variance;

  // Load weight matrix
  ar & w_;
  ar & w_delta_;
  ar & w_input_;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::
  save(std::ostream& os, ClassVersionType version)
{
  boost::archive::text_oarchive oa(os);
  save(oa, version);
  return true;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::
  load(std::istream& is, ClassVersionType version)
{
  boost::archive::text_iarchive ia(is);
  load(ia, version);
  return true;
}

template<typename ValueType,
         template<class> class NeuronTypeAtCompileTime,
         FFNN_SIZE_TYPE InputsAtCompileTime,
         FFNN_SIZE_TYPE OutputsAtCompileTime>
bool FullyConnected<ValueType, NeuronTypeAtCompileTime, InputsAtCompileTime, OutputsAtCompileTime>::setup()
{
  // Set biasing element
  (*Base::input_)(Base::input_->rows() - 1) = 1;

  // Allocate weighted-input vector
  w_input_.setZero(Base::output_dimension_, 1);

  // Set weight-delta matrix
  w_delta_.setZero(Base::output_dimension_, Base::input_dimension_);

  // Set random weight matrix
  w_.setRandom(Base::output_dimension_, Base::input_dimension_);
  w_ *= config_.weight_init_variance;

  // Initialize neurons
  neurons_.reserve(Base::output_dimension_);
  for (OffsetType idx = 0; idx < Base::output_dimension_; idx++)
  {
    neurons_.emplace_back(new NeuronTypeAtCompileTime<ValueType>());
  }

  FFNN_DEBUG_NAMED("layer::FullyConnected",
                   "<" <<
                   Base::id() <<
                   "> initialized as (in=" <<
                   (Base::input_dimension_ - 1) <<
                   ", out=" <<
                   Base::output_dimension_ <<
                   ") [with 1 biasing input]");
  return true;
}
}  // namespace layer
}  // namespace ffnn
