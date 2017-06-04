/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_SHAPE_H
#define FFNN_LAYER_INTERNAL_SHAPE_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

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
  FFNN_STATIC_ASSERT(std::is_integral<SizeType>::value, "SizeType must be an integer type.");
  FFNN_STATIC_ASSERT(std::is_signed<SizeType>::value, "SizeType must be a signed type.");

  SizeType height; ///< Height coordinate; expected to be greater than zero
  SizeType width;  ///< Width coordinate; expected to be greater than zero
  SizeType depth;  ///< Depth coordinate; expected to be greater than zero

  /**
   * @brief Default constructor
   */
  Shape() :
    height(Eigen::Dynamic),
    width(Eigen::Dynamic),
    depth(Eigen::Dynamic)
  {}

  /**
   * @brief Component constructor
   * @param height  height coordinate
   * @param width  width coordinate
   * @param depth  depth coordinate
   */
  explicit
  Shape(SizeType height, SizeType width = 1, SizeType depth = 1) :
    height(height),
    width(width),
    depth(depth)
  {}

  /**
   * @brief Returns product of all dimentions
   * @return <code>height * width * depth</code>
   */
  inline SizeType size() const
  {
    return multiply_if_not_dynamic_sizes(height, width, depth);
  }

  /**
   * @brief Checks if Shape object is valid
   * @retval true  if all coordinate fields represent sizes greater than zero
   * @retval false if any coordinate fields represent a dynamic (unassigned) size
   */
  inline bool valid() const
  {
    return size() > 0;
  }

  /**
   * @brief Casts Shape to a scalar value
   */
  operator SizeType() const { return size();}

  /**
   * @brief Assign Shape from a scalar value
   * @param size  interger size
   */
  Shape& operator=(SizeType size)
  {
    height = size;
    width  = 1;
    depth  = 1;
    return *this;
  }

  /**
   * @brief Assignment operator
   * @param other  Another Shape object
   */
  Shape& operator=(const Shape& other)
  {
    height = other.height;
    width  = other.width;
    depth  = other.depth;
    return *this;
  }

  /**
   * @brief Boost serializer
   */
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
