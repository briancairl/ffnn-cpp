/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_FILTER_BANK_H
#define FFNN_LAYER_INTERNAL_FILTER_BANK_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/layer/shape.h>
#include <ffnn/internal/traits.h>

namespace ffnn
{
namespace layer
{
namespace convolution
{
/**
 * @brief Sets all Filter kernels and scalar bias unit to zero
 * @param height  height of kernel matrix
 * @param widht  width of kernel matrix
 */
template<typename ValueType,
         ffnn::size_type HeightAtCompileTime = Eigen::Dynamic,
         ffnn::size_type WidthAtCompileTime  = Eigen::Dynamic,
         int DataOrdering = Eigen::ColMajor>
struct Filter :
  public std::vector<Eigen::Matrix<ValueType, HeightAtCompileTime, WidthAtCompileTime, DataOrdering>>
{
  /// Filter kernel matrix standardization
  typedef Eigen::Matrix<ValueType, HeightAtCompileTime, WidthAtCompileTime, DataOrdering> KernelType;

  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef ffnn::size_type SizeType;

  /// Offset type standardization
  typedef ffnn::offset_type OffsetType;

  /// Bias value
  ValueType bias;

  /**
   * @brief Default constructor
   */
  Filter() :
    bias(0.0)
  {}

  /**
   * @brief Sets all Filter kernels and scalar bias unit to zero
   * @param height  height of kernel matrix
   * @param widht  width of kernel matrix
   */
  void setZero(SizeType height = HeightAtCompileTime,
               SizeType width  = WidthAtCompileTime)
  {
    for (auto& filter : *this)
    {
      filter.kernel.setZero(height, width);
    }
    filter.bias = 0;
  }

  /**
   * @brief Scales kernels and bias value
   * @param scale  scalar value
   */
  void operator*=(ValueType scale)
  {
    for (auto& filter : *this)
    {
      filter.kernel *= scale;
    }
    this->bias *= scale;
  }

  /**
   * @brief In-place subtraction between two Filter objects
   * @param other  Filter object
   */
  void operator-=(const Filter& other)
  {
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
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel += other[idx].kernel;
    }
    this->bias += other.bias;
  }


  template<class Archive>
  void save(Archive & ar, const unsigned int version) const
  {
    SizeType n = this->size();

    ar & filter.bias;
    ar & n;
    for (const auto& filter : *this)
    {
      ar & filter.kernel;
    }
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version)
  {
    SizeType n;

    ar & filter.bias;
    ar & n;
    this->resize(n);
    for (SizeType idx = 0; idx < n; idx++)
    {
      ar & (*this)[idx].kernel;
    }
  }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int file_version)
  {
    boost::serialization::split_member(ar, *this, file_version);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(internal::traits::is_alignable_128<KernelType>::value);
};
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_FILTER_BANK_H
