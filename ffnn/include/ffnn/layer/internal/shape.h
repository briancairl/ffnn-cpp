/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_SHAPE_H
#define FFNN_LAYER_INTERNAL_SHAPE_H

// C++ Standard Library
#include <type_traits>

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>
#include <ffnn/internal/signature.h>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace layer
{
namespace internal
{

template<typename SizeType>
constexpr bool is_dynamic(SizeType n)
{
  return n == Eigen::Dynamic;
}

template<typename SizeType>
constexpr bool is_dynamic(SizeType n, SizeType m)
{
  return is_dynamic(n) || is_dynamic(m);
}

template<typename SizeType>
constexpr bool is_dynamic(SizeType n, SizeType m, SizeType l)
{
  return is_dynamic(n, m) || is_dynamic(l);
}

template<typename SizeType>
constexpr SizeType multiply_if_not_dynamic_sizes(SizeType n, SizeType m)
{
  return is_dynamic(n, m) ? Eigen::Dynamic : (n * m);
}

template<typename SizeType>
constexpr SizeType multiply_if_not_dynamic_sizes(SizeType n, SizeType m, SizeType l)
{
  return is_dynamic(n, m, l) ? Eigen::Dynamic : (n * m  * l);
}

template<typename SizeType>
struct Shape
{
  SizeType height;
  SizeType width;
  SizeType depth;

  Shape() :
    height(Eigen::Dynamic),
    width(Eigen::Dynamic),
    depth(Eigen::Dynamic)
  {}

  explicit
  Shape(SizeType height, SizeType width = 1, SizeType depth = 1) :
    height(height),
    width(width),
    depth(depth)
  {}

  inline SizeType size() const
  {
    return multiply_if_not_dynamic_sizes(height, width, depth);
  }

  inline bool valid() const
  {
    return size() > 0;
  }

  operator SizeType() const { return size(); }

  void operator=(SizeType count)
  {
    height = count;
    width = 1;
    depth = 1;
  }

  void operator=(const Shape& dim)
  {
    height = dim.height;
    width = dim.width;
    depth = dim.depth;
  }

  /// Serializer
  template<class Archive>
  void serialize(Archive & ar, const unsigned int file_version)
  {
    ar & height;
    ar & width;
    ar & depth;
  }
};

template<typename SizeType>
std::ostream& operator<<(std::ostream& os, const Shape<SizeType>& dim)
{
  if (dim.valid())
  {
    os << "<" << dim.height << " x " << dim.width << " x " << dim.depth << ">";
  }
  else
  {
    os << "<undefined>";
  }
  return os;
}
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_SHAPE_H
