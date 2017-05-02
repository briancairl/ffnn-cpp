/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_ALIGNED_TYPES_H
#define FFNN_ALIGNED_TYPES_H

// C++ Standard Library
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace aligned
{
template<typename ValueType>
struct Buffer :
  public std::vector<ValueType, Eigen::aligned_allocator<ValueType>>
{
  // Inherit base type assets
  using Base = std::vector<ValueType, Eigen::aligned_allocator<ValueType>>;
  using Base::Base;
};

template<typename MatrixType, typename ValueType, typename Stride>
struct MapBase;

template<typename MatrixType, typename Stride>
struct MapBase<MatrixType, float, Stride> :
  Eigen::Map<MatrixType, 16, Stride>
{
  // Inherit base type assets
  using Base = Eigen::Map<MatrixType, 16, Stride>;
  using Base::Base;

  // Scalar type standardization
  typedef double ScalarType;

  // Size type standardization
  typedef typename MatrixType::Index SizeType;

  MapBase(ScalarType* data, SizeType rows, SizeType cols) :
    Base(data, rows, cols)
  {}
  virtual ~MapBase() {}
};

template<typename MatrixType, typename Stride>
struct MapBase<MatrixType, double, Stride> :
  Eigen::Map<MatrixType, 32, Stride>
{
  // Inherit base type assets
  using Base = Eigen::Map<MatrixType, 32>;
  using Base::Base;

  // Scalar type standardization
  typedef double ScalarType;

  // Size type standardization
  typedef typename MatrixType::Index SizeType;

  MapBase(ScalarType* data, SizeType rows, SizeType cols) :
    Base(data, rows, cols)
  {}
  virtual ~MapBase() {}
};

/// Aligned mem-mapped matrix/vector type wrapper
template<typename MatrixType, typename Stride = Eigen::OuterStride<>>
struct Map :
  public MapBase<MatrixType, typename MatrixType::Scalar, Stride>
{
  // Inherit base type assets
  using Base = MapBase<MatrixType, typename MatrixType::Scalar, Stride>;
  using Base::Base;

  /// Shared resource standardization
  typedef boost::shared_ptr<Map> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const Map> ConstPtr;

  /**
   * @brief Creates a shared-resource pointer to a mapped vector
   * @param data  pointer to first element of a raw buffer
   * @param rows  number of rows represented in the buffer
   * @param cols  number of collumns represented in the buffer
   * @return shared-resource pointer
   */
  static typename Map<MatrixType, Stride>::Ptr
  create(typename MatrixType::Scalar* data,
         typename MatrixType::Index rows,
         typename MatrixType::Index cols = 1)
  {
    return boost::make_shared<Map<MatrixType>>(data, rows, cols);
  }
};
}  // namespace aligned
}  // namespace ffnn
#endif  // FFNN_ALIGNED_TYPES_H
