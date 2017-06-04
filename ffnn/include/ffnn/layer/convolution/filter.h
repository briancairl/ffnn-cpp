/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_FILTER_H
#define FFNN_LAYER_CONVOLUTION_FILTER_H

// C++ Standard Library
#include <array>
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/config/global.h>
#include <ffnn/internal/traits.h>

#include <ffnn/distribution/distribution.h>

#include <ffnn/layer/shape.h>
#include <ffnn/layer/convolution/sizing.h>
#include <ffnn/layer/convolution/filter/compile_time_options.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
/**
 * @brief Filter parameters to be use with a Convolution layer
 * @param ValueType scalar value type
 * @param Options  filter sizing and data-ordering information
 */
template<typename ValueType,
         typename Options    = typename filter::options<>,
         typename Extrinsics = typename filter::extrinsics<ValueType, Options>>
class Filter :
  public Extrinsics::FilterBaseType
{
  FFNN_ASSERT_NO_MODIFY_EXTRINSICS(filter);
public:
  /// Filter kernel matrix standardization
  typedef typename Extrinsics::KernelType KernelType;

  /// Bias value
  ValueType bias;

  /**
   * @brief Default constructor
   */
  template<bool T = Options::has_fixed_kernel_count>
  Filter(typename std::enable_if<T>::type* = nullptr) :
    bias(0)
  {}
  template<bool T = Options::has_fixed_kernel_count>
  Filter(size_type filter_count = Eigen::Dynamic,
         typename std::enable_if<!T>::type* = nullptr) :
    bias(0)
  {
    if (filter_count > 0)
    {
      this->resize(filter_count);
    }
  }

  /**
   * @brief Sets all Filter kernels and scalar bias unit to zero
   */
  void setZero()
  {
    for (auto& kernel : *this)
    {
      kernel.setZero();
    }
    bias = 0;
  }

  /**
   * @brief Sets all Filter kernels and scalar bias unit to zero
   * @param kernel_height  height of filter kernel
   * @param kernel_width  width of filter kernel
   * @param kernel_depth  depth of filter kernel
   * @param kernel_count  number of kernels
   */
  template<bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<T>::type
    setZero(size_type kernel_height,
            size_type kernel_width,
            size_type kernel_depth,
            size_type kernel_count)
  {
    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_width must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_depth must be positive");
    FFNN_ASSERT_MSG(kernel_count > 0,  "kernel_count must be positive");
    FFNN_ASSERT_MSG(kernel_count == Options::kernel_count,  "kernel_count is fixed");

    setZero();
  }
  template<bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<!T>::type
    setZero(size_type kernel_height,
            size_type kernel_width,
            size_type kernel_depth,
            size_type kernel_count)
  {
    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_width must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_depth must be positive");
    FFNN_ASSERT_MSG(kernel_count > 0,  "kernel_count must be positive");

    this->resize(kernel_count);
    const auto h = embed_dimension<Options::embedding_mode, ColEmbedding>(kernel_height, kernel_depth);
    const auto w = embed_dimension<Options::embedding_mode, RowEmbedding>(kernel_width,  kernel_depth);
    for (auto& kernel : *this)
    {
      kernel.setZero(h, w);
    }
    bias = 0;
  }

  /**
   * @brief Sets all Filter kernels and scalar bias unit to a random value
   *        drawn from a specified distribution
   */
  template<typename DistributionType>
  void setRandom(const DistributionType& distribution)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    for (auto& kernel : *this)
    {
      distribution::setRandom(kernel, distribution);
    }
    bias = distribution.generate();
  }

  /**
   * @brief Sets all Filter kernels and scalar bias unit to a random value
   *        drawn from a specified distribution
   * @param distribution
   * @param kernel_height  height of filter kernel
   * @param kernel_width  width of filter kernel
   * @param kernel_depth  depth of filter kernel
   * @param kernel_count  number of kernels
   */
  template<typename DistributionType,
           bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<T>::type
    setRandom(const DistributionType& distribution,
              size_type kernel_height,
              size_type kernel_width,
              size_type kernel_depth,
              size_type kernel_count)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_width must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_depth must be positive");
    FFNN_ASSERT_MSG(kernel_count > 0,  "kernel_count must be positive");
    FFNN_ASSERT_MSG(kernel_count == Options::kernel_count,  "kernel_count is fixed");

    setRandom(distribution);
  }
  template<typename DistributionType,
           bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<!T>::type
    setRandom(const DistributionType& distribution,
              size_type kernel_height,
              size_type kernel_width,
              size_type kernel_depth,
              size_type kernel_count)
  {
    static_assert(internal::traits::is_distribution<DistributionType>::value,
                  "[DistributionType] MUST FUFILL DISTRIBUTION CONCEPT REQUIREMENTS!");

    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_width must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_depth must be positive");
    FFNN_ASSERT_MSG(kernel_count > 0,  "kernel_count must be positive");

    this->resize(kernel_count);
    const auto h = embed_dimension<Options::embedding_mode, ColEmbedding>(kernel_height, kernel_depth);
    const auto w = embed_dimension<Options::embedding_mode, RowEmbedding>(kernel_width,  kernel_depth);
    for (auto& kernel : *this)
    {
      kernel.resize(h, w);
      distribution::setRandom(kernel, distribution);
    }
    bias = distribution.generate();
  }

  /**
   * @brief Scales kernels and bias value
   * @param scale  scalar value
   * @return *this
   */
  Filter& operator*=(ValueType scale)
  {
    for (auto& kernel : *this)
    {
      kernel *= scale;
    }
    this->bias *= scale;
    return *this;
  }

  /**
   * @brief In-place subtraction between two Filter objects
   * @param other  Filter object
   * @return *this
   */
  Filter& operator-=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx] -= other[idx];
    }
    this->bias -= other.bias;
    return *this;
  }

  /**
   * @brief In-place addition between two Filter objects
   * @param other  Filter object
   * @return *this
   */
  Filter& operator+=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx] += other[idx];
    }
    this->bias += other.bias;
    return *this;
  }

  /**
   * @brief In-place coefficient-wise multiplication between two Filter objects
   * @param other  Filter object
   * @return *this
   */
  Filter& operator*=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].array() *= other[idx].array();
    }
    this->bias *= other.bias;
    return *this;
  }

  /**
   * @brief In-place coefficient-wise division between two Filter objects
   * @param other  Filter object
   * @return *this
   */
  Filter& operator/=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].array() /= other[idx].array();
    }
    this->bias /= other.bias;
    return *this;
  }

  /**
   * @brief Returns reference <code>*this</code>
   * @note  Necessary to fufill <code>ParameterType</code> concept
   */
  Filter& array() { return *this;}

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
private:
  #include <ffnn/impl/layer/convolution/filter/serialization_class_definitions.hpp>
#endif
};
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_FILTER_H
