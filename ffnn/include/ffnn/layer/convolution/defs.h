/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_DEFS_H
#define FFNN_LAYER_CONVOLUTION_DEFS_H

namespace ffnn
{
namespace layer
{
namespace convolution
{
typedef enum
{
  RowEmbedding = 0, ///< Embed depth along filter matrix rows
  ColEmbedding = 1, ///< Embed depth along filter matrix cols
}
EmbeddingMode;

/**
 * @brief Selects size or depth-embedded size based on embedding mode
 * @param n  interger dimension
 * @param m  interger dimension
 * @retval (n*m) if (mode == ref) and sizes are not dynamic
 * @retval n if (mode != ref)
 */
template<EmbeddingMode mode, EmbeddingMode ref, typename SizeType>
constexpr SizeType embed_dimension(SizeType n, SizeType m)
{
  return (mode == ref) ? multiply_if_not_dynamic_sizes<SizeType>(n, m) : n;
}

/**
 * @brief Transforms shape dimensions to a depth-embedded representation
 * @param shape  original shape repsentation
 * @return depth-embedded version of shape
 */
template<EmbeddingMode mode, typename ShapeType>
ShapeType embed_shape_transform(const ShapeType& shape)
{
  const auto he = embed_dimension<mode, ColEmbedding>(shape.height, shape.depth);
  const auto we = embed_dimension<mode, RowEmbedding>(shape.width,  shape.depth);
  return ShapeType(he, we, 1);
}

/**
 * @brief Computes convolution output size
 * @param n  convolution input layer dimension
 * @param fn  filter dimension
 * @param stide  filter stride
 * @retval size if sizes are not dynamic
 */
template<typename SizeType>
constexpr SizeType output_dimension(SizeType n, SizeType fn, SizeType stride)
{
  return (is_dynamic<SizeType>(n)  ||
          is_dynamic<SizeType>(fn) ||
          is_dynamic<SizeType>(stride)) ? Eigen::Dynamic : ((n - fn) / stride + 1);
}
}  // namespace convolution
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_DEFS_H
