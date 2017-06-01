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
#include <ffnn/layer/shape.h>
#include <ffnn/layer/convolution/defs.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
/**
 * @brief Describes compile-time traits used to set up a Filter object
 * @param ValueType scalar value type
 * @param HeightAtCompileTime  height of the filter kernel
 * @param WidthAtCompileTime  width of the filter kernel
 * @param DepthAtCompileTime  depth of the filter kernel
 * @param KernelsAtCompileTime  number of kernels this filter will have
 */
template<typename ValueType,
         size_type HeightAtCompileTime = Eigen::Dynamic,
         size_type WidthAtCompileTime  = Eigen::Dynamic,
         size_type DepthAtCompileTime  = Eigen::Dynamic,
         size_type KernelsAtCompileTime = Eigen::Dynamic,
         EmbeddingMode Mode = ColEmbedding>
struct filter_traits
{
  /// Data embedding mode
  constexpr static EmbeddingMode embedding_mode = Mode;

  /// Depth embedded kernel height at compile-time
  constexpr static size_type kernel_height = embed_dimension<Mode, ColEmbedding>(HeightAtCompileTime, DepthAtCompileTime);

  /// Depth embedded kernel width at compile-time
  constexpr static size_type kernel_width  = embed_dimension<Mode, RowEmbedding>(WidthAtCompileTime,  DepthAtCompileTime);

  /// Number of filter kernels at compile-time
  constexpr static size_type kernel_count  = KernelsAtCompileTime;

  /// Data ordering at compile time
  /// @note Base on embedding_mode
  constexpr static size_type data_ordering = (Mode == ColEmbedding) ? Eigen::ColMajor : Eigen::RowMajor;

  /// <code>true</code> if number of kernels in the filter is determined at compile-time and is fixed
  constexpr static bool has_fixed_kernel_count  = (KernelsAtCompileTime > 0);

  /// Kernel type standardization
  typedef Eigen::Matrix<ValueType, kernel_height, kernel_width, data_ordering> KernelType;

  /// Base type standardization
  typedef typename std::conditional<
    has_fixed_kernel_count,
    std::array<KernelType, KernelsAtCompileTime>,
    typename std::conditional<
      internal::traits::is_alignable_128<KernelType>::value,
      std::vector<KernelType, Eigen::aligned_allocator<KernelType>>,
      std::vector<KernelType>
    >::type
  >::type BaseType;
};

/**
 * @brief Sets all Filter kernels and scalar bias unit to zero
 * @param ValueType scalar value type
 * @param FilterTraits  filter sizing and data-ordering information
 */
template<typename ValueType,
         typename FilterTraits = filter_traits<ValueType>>
struct Filter :
  public FilterTraits::BaseType
{
  /// Filter kernel matrix standardization
  typedef typename FilterTraits::KernelType KernelType;

  /// Bias value
  ValueType bias;

  /**
   * @brief Default constructor
   */
  template<bool T = FilterTraits::has_fixed_kernel_count>
  Filter(typename std::enable_if<T>::type* = nullptr) :
    bias(0.0)
  {}
  template<bool T = FilterTraits::has_fixed_kernel_count>
  Filter(size_type filter_count = Eigen::Dynamic,
         typename std::enable_if<!T>::type* = nullptr) :
    bias(0.0)
  {
    if (filter_count > 0)
    {
      this->resize(filter_count);
    }
  }

  /**
   * @brief Sets all Filter kernels and scalar bias unit to zero
   * @param kernel_height  height of filter kernel
   * @param kernel_width  width of filter kernel
   * @param kernel_depth  depth of filter kernel
   * @param kernel_count  number of kernels
   */
  template<bool T = FilterTraits::has_fixed_kernel_count>
  typename std::enable_if<T>::type
    setZero(size_type kernel_height,
            size_type kernel_width,
            size_type kernel_depth,
            size_type kernel_count)
  {
    FFNN_ASSERT_MSG(kernel_count == FilterTraits::kernel_count, "kernel_count is fixed");
    const auto he = embed_dimension<FilterTraits::embedding_mode, ColEmbedding>(kernel_height, kernel_depth);
    const auto we = embed_dimension<FilterTraits::embedding_mode, RowEmbedding>(kernel_width,  kernel_depth);
    for (auto& kernel : *this)
    {
      kernel.setZero(he, we);
    }
    bias = 0;
  }
  template<bool T = FilterTraits::has_fixed_kernel_count>
  typename std::enable_if<!T>::type
    setZero(size_type kernel_height,
            size_type kernel_width,
            size_type kernel_depth,
            size_type kernel_count)
  {
    this->resize(kernel_count);
    const auto he = embed_dimension<FilterTraits::embedding_mode, ColEmbedding>(kernel_height, kernel_depth);
    const auto we = embed_dimension<FilterTraits::embedding_mode, RowEmbedding>(kernel_width,  kernel_depth);
    for (auto& kernel : *this)
    {
      kernel.setZero(he, we);
    }
    bias = 0;
  }

  /**
   * @brief Scales kernels and bias value
   * @param scale  scalar value
   */
  void operator*=(ValueType scale)
  {
    for (auto& kernel : *this)
    {
      this->kernel *= scale;
    }
    this->bias *= scale;
  }

  /**
   * @brief In-place subtraction between two Filter objects
   * @param other  Filter object
   */
  void operator-=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel -= other[idx].kernel;
    }
    this->bias -= other.bias;
  }

  /**
   * @brief In-place addition between two Filter objects
   * @param other  Filter object
   */
  void operator+=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel += other[idx].kernel;
    }
    this->bias += other.bias;
  }

  /**
   * @brief In-place coefficient-wise multiplication between two Filter objects
   * @param other  Filter object
   */
  void operator*=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel.array() /= other[idx].kernel.array();
    }
    this->bias /= other.bias;
  }

  /**
   * @brief In-place coefficient-wise division between two Filter objects
   * @param other  Filter object
   */
  void operator/=(const Filter& other)
  {
    FFNN_ASSERT_MSG(this->size() == other.size(), "Filter sizes inconsistent");
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel.array() /= other[idx].kernel.array();
    }
    this->bias /= other.bias;
  }

  /**
   * @brief Returns reference <code>*this</code>
   * @note  Necessary to fufill <code>ParameterType</code> concept
   */
  Filter& array() { return *this;}

  /**
   * @brief Save serializer
   * @param ar  output archive
   * @param version  archive versioning information
   */
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const
  {
    size_type n = this->size();

    ar & bias;
    ar & n;
    for (const auto& kernel : *this)
    {
      ar & kernel;
    }
  }

  /**
   * @brief Load serializer
   * @param ar  input archive
   * @param version  archive versioning information
   * @note Statically sized version
   */
  template<class Archive, bool T = FilterTraits::has_fixed_kernel_count>
  typename std::enable_if<T>::type
    load(Archive & ar, const unsigned int version)
  {
    size_type n = this->size();

    ar & bias;
    ar & n ;
    for (size_type idx = 0; idx < n; idx++)
    {
      ar & (*this)[idx];
    }
  }
  template<class Archive, bool T = FilterTraits::has_fixed_kernel_count>
  typename std::enable_if<!T>::type
    load(Archive & ar, const unsigned int version)
  {
    size_type n;

    ar & bias;
    ar & n;

    this->resize(n);
    for (size_type idx = 0; idx < n; idx++)
    {
      ar & (*this)[idx];
    }
  }

  /**
   * @brief Serializer
   * @param ar  input/output archive
   * @param version  archive versioning information
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int file_version)
  {
    boost::serialization::split_member(ar, *this, file_version);
  }
};
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_FILTER_H
