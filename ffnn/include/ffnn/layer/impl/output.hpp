/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */
// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
namespace layer
{
template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<ValueType, NetworkOutputsAtCompileTime>::Output(SizeType network_output_dim) :
  Base(network_output_dim, 0)
{}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
Output<ValueType, NetworkOutputsAtCompileTime>::~Output()
{}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
bool Output<ValueType, NetworkOutputsAtCompileTime>::initialize()
{
  // Abort if layer is already initialized
  if (Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Layer", "<" << Base::id() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  SizeType input_count = Base::countInputs();
  {
    // Validate network input count
    FFNN_ASSERT_MSG (NetworkOutputsAtCompileTime < 0 || input_count == NetworkOutputsAtCompileTime,
                     "(NetworkOutputsAtCompileTime != `resolved input size`) for fixed-size layer.");
  
    // Set network input count
    Base::input_dimension_ = input_count;
  }

  // Do basic initialization and connect last hidden layer
  if (Base::initialize() && Base::connectInputLayers() == input_count)
  {
    FFNN_DEBUG_NAMED("layer::Output",
                     "<" << Base::id() <<
                     "> initialized as network output (net-out=" <<
                     Base::input_dimension_ << ")");
    return true;
  }

  // Error initializing
  Base::initialized_ = false;
  FFNN_ERROR_NAMED("layer::Output", "< " << Base::id() << "> failed to initialize.");
  return false;
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
typename Output<ValueType, NetworkOutputsAtCompileTime>::OffsetType
Output<ValueType, NetworkOutputsAtCompileTime>::connectToForwardLayer(const Base& next, OffsetType offset)
{
  return 0; /* do nothing */
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkOutputType>
void Output<ValueType, NetworkOutputsAtCompileTime>::operator>>(NetworkOutputType& output)
{
  // Check output data size
  FFNN_ASSERT_MSG(output.size() == Base::input_dimension_,
                  "Output object size does not match expected network output size.");

  // Copy output data from last network layer
  std::memcpy(const_cast<ValueType*>(output.data()),
              const_cast<ValueType*>(Base::input_buffer_.data()),
              Base::input_buffer_.size() * sizeof(ValueType));
}

template<typename ValueType, FFNN_SIZE_TYPE NetworkOutputsAtCompileTime>
template<typename NetworkTargetType>
void Output<ValueType, NetworkOutputsAtCompileTime>::operator<<(const NetworkTargetType& target)
{
  // Check target data size
  FFNN_ASSERT_MSG(target.size() == Base::input_dimension_,
                  "Target object size does not match expected network output size.");

  // Mapped vector type aliases
  using Vector = Eigen::Matrix<ValueType, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using MapVec = typename Mapped<Vector>::Type;

  // Compute network error
  MapVec(const_cast<ValueType*>(Base::backward_error_buffer_.data()), Base::backward_error_buffer_.size(), 1) =
    MapVec(const_cast<ValueType*>(Base::input_buffer_.data()), Base::input_buffer_.size(), 1) -
    MapVec(const_cast<ValueType*>(target.data()), target.size(), 1);
}
}  // namespace layer
}  // namespace ffnn
