/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_SHAPE_H
#define FFNN_LAYER_INTERNAL_SHAPE_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Checks if size represents a dynamic size
 * @param n  interger dimension
 * @retval true if dimension represents a dynamic size
 * @retval false otherwise
 */
template<typename SizeType>
constexpr bool is_dynamic(SizeType n)
{
  return n == Eigen::Dynamic;
}

/**
 * @brief Checks if a pair of sizes is dynamic
 * @param n  interger dimension
 * @param m  interger dimension
 * @retval true if either dimension represents a dynamic size
 * @retval false otherwise
 */
template<typename SizeType>
constexpr bool is_dynamic(SizeType n, SizeType m)
{
  return is_dynamic(n) || is_dynamic(m);
}

/**
 * @brief Checks if a triplet of sizes is dynamic
 * @param n  interger dimension
 * @param m  interger dimension
 * @param l  interger dimension
 * @retval true if any dimension represents a dynamic size
 * @retval false otherwise
 */
template<typename SizeType>
constexpr bool is_dynamic(SizeType n, SizeType m, SizeType l)
{
  return is_dynamic(n, m) || is_dynamic(l);
}

/**
 * @brief Multiplies a pair of sizes if all sizes are static
 * @param n  interger dimension
 * @param m  interger dimension
 * @retval (n*m) if sizes are not dynamic
 * @retval Eigen::Dynamic otherwise
 */
template<typename SizeType>
constexpr SizeType multiply_if_not_dynamic_sizes(SizeType n, SizeType m)
{
  return is_dynamic(n, m) ? Eigen::Dynamic : (n * m);
}

/**
 * @brief Multiplies a triplet of sizes if all sizes are static
 * @param n  interger dimension
 * @param m  interger dimension
 * @param l  interger dimension
 * @retval (n*m*l) if sizes are not dynamic
 * @retval Eigen::Dynamic otherwise
 */
template<typename SizeType>
constexpr SizeType multiply_if_not_dynamic_sizes(SizeType n, SizeType m, SizeType l)
{
  return is_dynamic(n, m, l) ? Eigen::Dynamic : (n * m  * l);
}

/**
 * @brief Shape structure representing the dimensions of an object with volume
 */
template<typename SizeType>
struct Shape
{
  static_assert(std::is_integral<SizeType>::value, "SizeType must be an integer type.");
  static_assert(std::is_signed<SizeType>::value, "SizeType must be a signed type.");

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
    width  = 1;
    depth  = 1;
  }

  void operator=(const Shape& dim)
  {
    height = dim.height;
    width  = dim.width;
    depth  = dim.depth;
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
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_SHAPE_H
