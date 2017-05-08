/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_SHAPE_H
#define FFNN_LAYER_INTERNAL_SHAPE_H

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
#define IS_DYNAMIC(x) (x == Eigen::Dynamic)
#define IS_DYNAMIC_TRIPLET(n, m, l) (IS_DYNAMIC(n) || IS_DYNAMIC(m) || IS_DYNAMIC(l))
#define PROD_IF_STATIC_TRIPLET(n, m, l) (IS_DYNAMIC_TRIPLET(n, m, l) ? Eigen::Dynamic : (n*m*l))

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
    return PROD_IF_STATIC_TRIPLET(height, width, depth);
  }

  inline bool valid() const
  {
    return PROD_IF_STATIC_TRIPLET(height, width, depth) > 0;
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
  if ((dim.height * dim.width * dim.depth) > 0)
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
