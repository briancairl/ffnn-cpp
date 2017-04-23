/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_MAPPED_H
#define FFNN_MAPPED_H

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
/// Aligned mem-mapped matrix/vector type wrapper
template<typename MatrixType>
struct Mapped
{
#ifndef FFNN_DONT_ALIGN
  /// Mapped-vector result type
  typedef Eigen::Map<MatrixType, 32> Type;
#else
  /// Mapped-vector result type
  typedef Eigen::Map<MatrixType> Type;
#endif

  /// Real value type standardization
  typedef typename MatrixType::Scalar ValueType;

  /// Size type standardization
  typedef typename MatrixType::Index SizeType;

  /// Shared-resrouce type standardization
  typedef boost::shared_ptr<Type> Ptr;

  /**
   * @brief Creates a shared-resource pointer to a mapped vector
   * @param data  pointer to first element of a raw buffer
   * @param nrows  number of rows represented in the buffer
   * @param ncols  number of collumns represented in the buffer
   * @return shared-resource pointer
   */
  static Ptr create(ValueType* data, SizeType nrows, SizeType ncols = 1)
  {
    return boost::make_shared<Type>(data, nrows, ncols);
  }
};
}  // namespace ffnn
#endif  // FFNN_MAPPED_H
