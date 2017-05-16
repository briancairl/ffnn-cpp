/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_DEFS_H
#define FFNN_LAYER_CONVOLUTION_DEFS_H

// FFNN (internal)
#include <ffnn/layer/internal/shape.h>

namespace ffnn
{
namespace layer
{
typedef enum
{
  RowEmbedding = 0, ///< Embed depth along filter matrix rows
  ColEmbedding = 1, ///< Embed depth along filter matrix cols
}
EmbeddingMode;

template<EmbeddingMode mode, EmbeddingMode ref, typename SizeType>
constexpr SizeType embed_dimension(SizeType n, SizeType m)
{
  return (mode == ref) ? internal::multiply_if_not_dynamic_sizes<SizeType>(n, m) : n;
}

template<typename SizeType>
constexpr SizeType output_dimension(SizeType n, SizeType fn, SizeType stride)
{
  return internal::is_dynamic<SizeType>(n) ? Eigen::Dynamic : ((n - fn) / stride + 1);
}
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_CONVOLUTION_DEFS_H
