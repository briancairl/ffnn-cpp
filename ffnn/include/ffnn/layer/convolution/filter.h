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
namespace filter
{
/**
 * @brief Describes compile-time options and extrinsic parameters used to set up a Filter object
 * @param HeightAtCompileTime  height of the filter kernel
 * @param WidthAtCompileTime  width of the filter kernel
 * @param DepthAtCompileTime  depth of the filter kernel
 * @param KernelCountAtCompileTime  number of kernels this filter will have
 */
template<size_type HeightAtCompileTime = Eigen::Dynamic,
         size_type WidthAtCompileTime  = Eigen::Dynamic,
         size_type DepthAtCompileTime  = Eigen::Dynamic,
         size_type KernelCountAtCompileTime = Eigen::Dynamic,
         EmbeddingMode Mode = ColEmbedding>
struct options
{
  /// Data embedding mode
  constexpr static EmbeddingMode embedding_mode = Mode;

  /// Kernel height at compile-time
  constexpr static size_type kernel_height = HeightAtCompileTime;

  /// Kernel width at compile-time
  constexpr static size_type kernel_width = WidthAtCompileTime;

  /// Kernel depth at compile-time
  constexpr static size_type kernel_depth = DepthAtCompileTime;

  /// Number of filter kernels at compile-time
  constexpr static size_type kernel_count = KernelCountAtCompileTime;

  /// Depth embedded kernel height at compile-time
  constexpr static size_type embedded_kernel_height = embed_dimension<embedding_mode, ColEmbedding>(kernel_height, kernel_depth);

  /// Depth embedded kernel width at compile-time
  constexpr static size_type embedded_kernel_width = embed_dimension<embedding_mode, RowEmbedding>(kernel_width, kernel_depth);

  /// <code>true</code> if number of kernels in the filter is determined at compile-time and is fixed
  constexpr static bool has_fixed_kernel_count = (KernelCountAtCompileTime > 0);
};

/**
 * @brief Describes types based on compile-time options
 */
template<typename ValueType,
         typename Options>
struct extrinsics
{
  /// Kernel type standardization
  typedef Eigen::Matrix<
    ValueType,
    Options::embedded_kernel_height,
    Options::embedded_kernel_width,
    embed_data_order<Options::embedding_mode>()
  > KernelType;

  /// Base type standardization
  typedef typename std::conditional<
    Options::has_fixed_kernel_count,
    std::array<KernelType, Options::kernel_count>,
    typename std::conditional<
      internal::traits::is_alignable_128<KernelType>::value,
      std::vector<KernelType, Eigen::aligned_allocator<KernelType>>,
      std::vector<KernelType>
    >::type
  >::type FilterBaseType;
};
}  // namespace filter

/**
 * @brief Filter parameters to be use with a Convolution layer
 * @param ValueType scalar value type
 * @param Options  filter sizing and data-ordering information
 */
template<typename ValueType,
         typename Options    = typename filter::options<>,
         typename Extrinsics = typename filter::extrinsics<ValueType, Options>>
struct Filter :
  public Extrinsics::FilterBaseType
{
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
   * @param kernel_height  height of filter kernel
   * @param kernel_width  width of filter kernel
   * @param kernel_depth  depth of filter kernel
   * @param kernel_count  number of kernels
   */
  template<bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<T>::type
    setZero(size_type kernel_height = Options::kernel_height,
            size_type kernel_width  = Options::kernel_width,
            size_type kernel_depth  = Options::kernel_depth,
            size_type kernel_count  = Options::kernel_count)
  {
    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_count > 0,  "kernel_count must be positive");
    FFNN_ASSERT_MSG(kernel_count == Options::kernel_count,  "kernel_count is fixed");
    const auto h = embed_dimension<Options::embedding_mode, ColEmbedding>(kernel_height, kernel_depth);
    const auto w = embed_dimension<Options::embedding_mode, RowEmbedding>(kernel_width,  kernel_depth);
    for (auto& kernel : *this)
    {
      kernel.setZero(h, w);
    }
    bias = 0;
  }
  template<bool T = Options::has_fixed_kernel_count>
  typename std::enable_if<!T>::type
    setZero(size_type kernel_height = Options::kernel_height,
            size_type kernel_width  = Options::kernel_width,
            size_type kernel_depth  = Options::kernel_depth,
            size_type kernel_count  = Options::kernel_count)
  {
    FFNN_ASSERT_MSG(kernel_height > 0, "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_width > 0,  "kernel_height must be positive");
    FFNN_ASSERT_MSG(kernel_depth > 0,  "kernel_height must be positive");
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
  template<class Archive, bool T = Options::has_fixed_kernel_count>
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
  template<class Archive, bool T = Options::has_fixed_kernel_count>
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
