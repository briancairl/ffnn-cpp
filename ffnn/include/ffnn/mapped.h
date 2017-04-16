/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_MAPPED_H
#define FFNN_MAPPED_H

// C++ Standard Library
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
/// Mapped-Matrix type wrapper
template<typename MatrixType>
struct Mapped
{
  typedef boost::shared_ptr<Eigen::Map<MatrixType>> Ptr;

  typedef Eigen::Map<MatrixType> Type;

  template<typename ArgPointerType>
  static Ptr create(ArgPointerType data,
                    typename MatrixType::Index nrows,
                    typename MatrixType::Index ncols = 1)
  {
    return boost::make_shared<Type>(const_cast<typename MatrixType::Scalar*>(data), nrows, ncols);
  }

  static Ptr create(const std::vector<typename MatrixType::Scalar>& v)
  {
    return create(v.data(), v.size(), 1UL);
  }
};
}  // namespace ffnn
#endif  // FFNN_MAPPED_H
