/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_DIMENSIONS_H
#define FFNN_LAYER_INTERNAL_DIMENSIONS_H

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
struct Dimensions
{
  FFNN_REGISTER_SERIALIZABLE(Dimensions)

  SizeType height;
  SizeType width;
  SizeType depth;

  Dimensions() :
    height(Eigen::Dynamic),
    width(Eigen::Dynamic),
    depth(Eigen::Dynamic)
  {}

  explicit
  Dimensions(SizeType height, SizeType width = 1, SizeType depth = 1) :
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

  void operator=(const Dimensions& dim)
  {
    height = dim.height;
    width = dim.width;
    depth = dim.depth;
  }

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const
  {
    ffnn::io::signature::apply<Dimensions<SizeType>>(ar);
    ar & height;
    ar & width;
    ar & depth;
  }

  /// Load serializer
  void load(InputArchive& ar, VersionType version)
  {
    ffnn::io::signature::check<Dimensions<SizeType>>(ar);
    ar & height;
    ar & width;
    ar & depth;
  }
};

template<typename SizeType>
std::ostream& operator<<(std::ostream& os, const Dimensions<SizeType>& dim)
{
  os << "<" << dim.height << " x " << dim.width << " x " << dim.depth << ">";
  return os;
}
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_DIMENSIONS_H
