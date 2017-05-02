/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H
#define FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H

// C++ Standard Library
#include <vector>

// FFNN
#include <ffnn/layer/hidden.h>

namespace ffnn
{
namespace layer
{
#define IS_DYNAMIC(x) (x == Eigen::Dynamic)
#define IS_DYNAMIC_PAIR(n, m) (IS_DYNAMIC(n) || IS_DYNAMIC(m))
#define IS_DYNAMIC_TRIPLET(n, m, l) (IS_DYNAMIC(n) || IS_DYNAMIC(m) || IS_DYNAMIC(l))
#define PROD_IF_STATIC_PAIR(n, m) (IS_DYNAMIC_PAIR(n, m) ? Eigen::Dynamic : (n*m))
#define PROD_IF_STATIC_TRIPLET(n, m, l) (IS_DYNAMIC_TRIPLET(n, m, l) ? Eigen::Dynamic : (n*m*l))
#define RECEPTIVE_VOLUME_INPUT_SIZE (PROD_IF_STATIC_TRIPLET(HeightAtCompileTime, WidthAtCompileTime, DepthAtCompileTime))

template <typename ValueType,
          FFNN_SIZE_TYPE FilterCount,
          FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE EmbedAlongColumns = true>
class ReceptiveVolume
{
public:
  typedef typename Base::SizeType SizeType;

  typedef typename Base::OffsetType OffsetType;

  typedef Eigen::Matrix<ValueType,
                        EmbedAlongColumns ? PROD_IF_STATIC_PAIR(HeightAtCompileTime, DepthAtCompileTime) : HeightAtCompileTime,
                        !EmbedAlongColumns ? PROD_IF_STATIC_PAIR(WidthAtCompileTime, DepthAtCompileTime) : WidthAtCompileTime,
                        Eigen::ColMajor> Kernel;

  typedef std::array<Kernel, FilterCount> FilterBank;

private:
  FilterBank filter_bank_;

  ReceptiveVolume() :
    Base(RECEPTIVE_VOLUME_INPUT_SIZE, FilterCount)
  {}

  template<typename InputBlockType, typename OutputBlockType>
  bool forward(const InputBlockType& input, OutputBlockType& output)
  {
    for (OffsetType idx = 0; idx < FilterCount; idx++)
    {
      output(idx) = (input.array() * filter_bank_[idx].array()).sum();
    }
    return true;
  }

  SizeType depth() const
  {
    return FilterCount;
  }
};
#undef IS_DYNAMIC
#undef IS_DYNAMIC_PAIR
#undef IS_DYNAMIC_TRIPLET
#undef PROD_IF_STATIC_PAIR
#undef PROD_IF_STATIC_TRIPLET
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_INTERNAL_RECEPTIVE_VOLUME_H
