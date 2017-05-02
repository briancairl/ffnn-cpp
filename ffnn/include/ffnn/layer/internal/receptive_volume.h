/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H
#define FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H

// C++ Standard Library
#include <vector>

namespace ffnn
{
namespace layer
{
namespace internal
{
#define IS_DYNAMIC(x) (x == Eigen::Dynamic)
#define IS_DYNAMIC_PAIR(n, m) (IS_DYNAMIC(n) || IS_DYNAMIC(m))
#define IS_DYNAMIC_TRIPLET(n, m, l) (IS_DYNAMIC(n) || IS_DYNAMIC(m) || IS_DYNAMIC(l))
#define PROD_IF_STATIC_PAIR(n, m) (IS_DYNAMIC_PAIR(n, m) ? Eigen::Dynamic : (n*m))
#define PROD_IF_STATIC_TRIPLET(n, m, l) (IS_DYNAMIC_TRIPLET(n, m, l) ? Eigen::Dynamic : (n*m*l))
#define RECEPTIVE_VOLUME_INPUT_SIZE (PROD_IF_STATIC_TRIPLET(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime))

enum EmbeddingMode
{
  RowEmbedding = 0,
  ColEmbedding = 1,
};

template <typename ValueType,
          FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterCountAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE EmbeddingMode = ColEmbedding>
class ReceptiveVolume
{
public:
  typedef typename Base::SizeType SizeType;

  typedef typename Base::OffsetType OffsetType;

  typedef Eigen::Matrix<ValueType,
                        EmbeddingMode == ColEmbedding ? PROD_IF_STATIC_PAIR(HeightAtCompileTime, DepthAtCompileTime) : HeightAtCompileTime,
                        EmbeddingMode == RowEmbedding ? PROD_IF_STATIC_PAIR(WidthAtCompileTime,  DepthAtCompileTime) : WidthAtCompileTime,
                        Eigen::ColMajor> Kernel;

  typedef Eigen::Matrix<ValueType, FilterCountAtCompileTime, 1, Eigen::ColMajor> BiasVector;
          
  typedef std::vector<Kernel> FilterBank;

  ReceptiveVolume()
  {}

  template<typename InputBlockType, typename OutputBlockType>
  bool forward(const InputBlockType& input, OutputBlockType& output)
  {
    for (size_t idx = 0; idx < filter_bank_.size(); idx++)
    {
      output(idx) = (input.array() * filter_bank_[idx].array()).sum() + b_(idx);
    }
    return true;
  }

private:
  FilterBank filter_bank_;
          
  BiasVector b_;
};
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H
