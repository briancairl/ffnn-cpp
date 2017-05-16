/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_FILTER_BANK_H
#define FFNN_LAYER_INTERNAL_FILTER_BANK_H

// C++ Standard Library
#include <vector>
#include <type_traits>

// FFNN (internal)
#include <ffnn/layer/internal/shape.h>
#include <ffnn/layer/internal/interface.h>

namespace ffnn
{
namespace layer
{
namespace internal
{

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
         int Option = Eigen::ColMajor>
struct Filter
{
  /// Filter kernel matrix standardization
  typedef Eigen::Matrix<ValueType, HeightAtCompileTime, WidthAtCompileTime, Option> KernelMatrixType;

  /// Kernel matrix
  KernelMatrixType kernel;

  /// Biasing scalar
  ValueType bias;

  /// Scalar type standardization
  typedef ValueType ScalarType;

  // Size type standardization
  typedef typename KernelMatrixType::Index SizeType;

  // Index type standardization
  typedef typename KernelMatrixType::Index OffsetType;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(internal::is_alignable_128<KernelMatrixType>::value);
};

template<typename ValueType,
         FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
         FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
         int Option = Eigen::ColMajor>
struct FilterBank :
  public std::vector<Filter<ValueType, HeightAtCompileTime, WidthAtCompileTime, Option>>
{
  /// Filter kernel matrix standardization
  typedef Filter<ValueType, HeightAtCompileTime, WidthAtCompileTime, Option> FilterType;

  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef typename FilterType::SizeType SizeType;

  /// Offset type standardization
  typedef typename FilterType::OffsetType OffsetType;

  FilterBank()
  {}

  void setZero(SizeType height, SizeType width)
  {
    for (auto& filter : *this)
    {
      filter.kernel.setZero(height, width);
      filter.bias = 0;
    }
  }

  void operator*=(ValueType scale)
  {
    for (auto& filter : *this)
    {
      filter.kernel *= scale;
      filter.bias *= scale;
    }
  }

  void operator-=(const FilterBank& other)
  {
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel -= other[idx].kernel;
      (*this)[idx].bias -= other[idx].bias;
    }
  }

  void operator+=(const FilterBank& other)
  {
    for (size_t idx = 0UL; idx < this->size(); idx++)
    {
      (*this)[idx].kernel += other[idx].kernel;
      (*this)[idx].bias += other[idx].bias;
    }
  }


  template<class Archive>
  void save(Archive & ar, const unsigned int version) const
  {
    SizeType n = this->size();

    ar & n;
    for (const auto& filter : *this)
    {
      ar & filter.kernel;
      ar & filter.bias;
    }
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version)
  {
    SizeType n;

    ar & n;
    this->resize(n);
    for (SizeType idx = 0; idx < n; idx++)
    {
      ar & (*this)[idx].kernel;
      ar & (*this)[idx].bias;
    }
  }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int file_version)
  {
    boost::serialization::split_member(ar, *this, file_version);
  }
};
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_FILTER_BANK_H
