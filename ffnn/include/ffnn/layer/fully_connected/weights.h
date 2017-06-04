/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_H
#define FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/internal/traits.h>
#include <ffnn/distribution/distribution.h>
#include <ffnn/layer/shape.h>
#include <ffnn/layer/fully_connected/weights/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace fully_connected
{
/**
 * @brief Weights parameters to be use with a FullyConnected layer
 * @param ValueType scalar value type
 * @param Options  weights sizing and data-ordering information
 */
template<typename ValueType,
         typename Options    = typename weights::options<>,
         typename Extrinsics = typename weights::extrinsics<ValueType, Options>>
class Weights
{
  FFNN_ASSERT_NO_MODIFY_EXTRINSICS(weights);
public:
  /// Weights kernel matrix standardization
  typedef typename Extrinsics::WeightBlockType WeightBlockType;

  /// Bias kernel matrix standardization
  typedef typename Extrinsics::BiasBlockType BiasBlockType;

  // Connection weights
  WeightBlockType weights;

  // Connection biases
  BiasBlockType biases;

  /**
   * @brief Default constructor
   */
  Weights() :
    bias(0)
  {}

  /**
   * @brief Sets all Weights kernels and biases to zero
   */
  void setZero()
  {
    weights.setZero();
    biases.setZero();
  }

  /**
   * @brief Sets all Weights kernels and scalar bias unit to zero
   * @param n_inputs  number of inputs to a connected layer
   * @param n_outputs  number of outputs from a connected layer
   */
  template<bool T = Options::has_fixed_sizes>
  typename std::enable_if<T>::type
    setZero(size_type n_inputs, size_type n_outputs)
  {
    FFNN_ASSERT_MSG(n_inputs > 0,  "n_inputs must be positive");
    FFNN_ASSERT_MSG(n_outputs > 0, "n_outputs must be positive");

    setZero();
  }
  template<bool T = Options::has_fixed_sizes>
  typename std::enable_if<!T>::type
    setZero(size_type n_inputs, size_type n_outputs)
  {
    FFNN_ASSERT_MSG(n_inputs > 0,  "n_inputs must be positive");
    FFNN_ASSERT_MSG(n_outputs > 0, "n_outputs must be positive");

    weights.setZero(n_outputs, n_inputs);
    biases.setZero(n_outputs);
  }

  /**
   * @brief Sets all Weights kernels and scalar bias unit to a random value
   *        drawn from a specified distribution
   */
  template<typename DistributionType>
  void setRandom(const DistributionType& distribution)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    distribution::setRandom(weights, distribution);
    distribution::setRandom(biases, distribution);
  }

  /**
   * @brief Sets all Weights kernels and scalar bias unit to a random value
   *        drawn from a specified distribution
   * @param distribution
   * @param n_inputs  number of inputs to a connected layer
   * @param n_outputs  number of outputs from a connected layer
   */
  template<typename DistributionType,
           bool T = Options::has_fixed_sizes>
  typename std::enable_if<T>::type
    setRandom(const DistributionType& distribution,
              size_type n_inputs,
              size_type n_outputs)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    FFNN_ASSERT_MSG(n_inputs > 0,  "n_inputs must be positive");
    FFNN_ASSERT_MSG(n_outputs > 0, "n_outputs must be positive");

    setRandom(distribution);
  }
  template<typename DistributionType,
           bool T = Options::has_fixed_sizes>
  typename std::enable_if<!T>::type
    setRandom(const DistributionType& distribution,
              size_type n_inputs,
              size_type n_outputs)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    FFNN_ASSERT_MSG(n_inputs > 0,  "n_inputs must be positive");
    FFNN_ASSERT_MSG(n_outputs > 0, "n_outputs must be positive");

    weights.resize(n_outputs, n_inputs);
    biases.resize(n_outputs);

    setRandom(distribution);
  }

  /**
   * @brief Scales kernels and bias value
   * @param scale  scalar value
   * @return *this
   */
  Weights& operator*=(ValueType scale)
  {
    weights *= scale;
    bias *= scale;
    return *this;
  }

  /**
   * @brief In-place subtraction between two Weights objects
   * @param other  Weights object
   * @return *this
   */
  Weights& operator-=(const Weights& other)
  {
    weights -= other.weights;
    biases -= other.biases;
    return *this;
  }

  /**
   * @brief In-place addition between two Weights objects
   * @param other  Weights object
   * @return *this
   */
  Weights& operator+=(const Weights& other)
  {
    weights += other.weights;
    biases += other.biases;
    return *this;
  }

  /**
   * @brief In-place coefficient-wise multiplication between two Weights objects
   * @param other  Weights object
   * @return *this
   */
  Weights& operator*=(const Weights& other)
  {
    weights.array() *= other.weights.array();
    biases.array() *= other.biases.array();
    return *this;
  }

  /**
   * @brief In-place coefficient-wise division between two Weights objects
   * @param other  Weights object
   * @return *this
   */
  Weights& operator/=(const Weights& other)
  {
    weights.array() /= other.weights.array();
    biases.array() /= other.biases.array();
    return *this;
  }

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
private:
  #include <ffnn/impl/layer/fully_connected/weights/serialization_class_definitions.hpp>
#endif
};
}  // namespace fully_connected
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_FULLY_CONNECTED_WEIGHTS_H
