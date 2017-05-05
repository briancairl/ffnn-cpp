/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_FULLY_CONNECTED_H
#define FFNN_LAYER_FULLY_CONNECTED_H

// Boost
#include "boost/multi_array.hpp"

// FFNN Layer
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/receptive_volume.h>
#include <ffnn/neuron/neuron.h>

// FFNN Optimization
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace layer
{
#define CONVOLUTION_OUTPUT_HEIGHT ((HeightAtCompileTime - FilterHeightAtCompileTime) / Stride + 1)
#define CONVOLUTION_OUTPUT_WIDTH  ((WidthAtCompileTime - FilterWidthAtCompileTime) / Stride + 1)

#define CONVOLUTION_BASE_TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  CONVOLUTION_OUTPUT_HEIGHT,\
  CONVOLUTION_OUTPUT_WIDTH

#define CONVOLUTION_VOLUME_TARGS\
  ValueType,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterCountAtCompileTime,\
  EmbeddingMode

#define CONVOLUTION_TARGS\
  ValueType,\
  HeightAtCompileTime,\
  WidthAtCompileTime,\
  DepthAtCompileTime,\
  FilterHeightAtCompileTime,\
  FilterWidthAtCompileTime,\
  FilterCountAtCompileTime,\
  Stride,\
  EmbeddingMode

/**
 * @brief A convolution layer
 */
template <typename ValueType,
          FFNN_SIZE_TYPE HeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE WidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE DepthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterHeightAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterWidthAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE FilterCountAtCompileTime = Eigen::Dynamic,
          FFNN_SIZE_TYPE Stride = 1,
          FFNN_SIZE_TYPE EmbeddingMode = ColEmbedding>
class Convolution :
  public Hidden<CONVOLUTION_BASE_TARGS>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType,
                      HeightAtCompileTime,
                      WidthAtCompileTime,
                      CONVOLUTION_OUTPUT_HEIGHT,
                      CONVOLUTION_OUTPUT_WIDTH>;

  /// Self type alias
  using Self = Convolution<CONVOLUTION_TARGS>;

  /// Scalar type standardization
  typedef typename Base::ScalarType ScalarType;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  /// Dimension type standardization
  typedef typename Base::DimType DimType;

  /// Receptive-volume type standardization
  typedef ReceptiveVolume<CONVOLUTION_VOLUME_TARGS> ReceptiveVolumeType;

  /// Recptive-volume bank standardization
  typedef boost::multi_array<typename ReceptiveVolumeType::Ptr, 2> ReceptiveVolumeBankType;

  /// Layer optimization type standardization
  typedef optimizer::Optimizer<Self> Optimizer;

  /// Configuration struct type alias
  typedef typename ReceptiveVolumeType::Parameters Parameters;

  /**
   * @brief Setup constructor
   * @param output_size  number of layer outputs
   * @param config  layer configuration struct
   */
  explicit
  Convolution() {};
  virtual ~Convolution() {};

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize()
  {
    receptors_.resize(boost::extents[CONVOLUTION_OUTPUT_HEIGHT][CONVOLUTION_OUTPUT_WIDTH]);

    Base::initialized_ = true;
    for (SizeType idx = 0; idx < CONVOLUTION_OUTPUT_HEIGHT; idx++)
    {
      for (SizeType jdx = 0; jdx < CONVOLUTION_OUTPUT_WIDTH; jdx++)
      {
        receptors_[idx][jdx] = boost::make_shared<ReceptiveVolumeType>();
        Base::initialized_ |= receptors_[idx][jdx]->initialize();

        FFNN_ASSERT_MSG(Base::initialized_, "Failed to initialize rececptor.");
      }
    }
    return Base::isInitialized();
  }

  /**
   * @brief Performs forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  //virtual bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   * @warning Does not apply layer weight updates
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  //virtual bool backward();

  /**
   * @brief Applies accumulated layer weight updates computed during optimization
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   * @warning Will throw if an optimizer has not been associated with this layer
   * @see setOptimizer
   */
  //virtual bool update();

  /**
   * @brief Reset weights and biases
   */
  //void reset();

  inline const ReceptiveVolumeBankType& getReceptiveVolumes() const
  {
    return receptors_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(Convolution)

  // /// Save serializer
  // void save(OutputArchive& ar, VersionType version) const;

  // /// Load serializer
  // void load(InputArchive& ar, VersionType version);

private:
  //FFNN_REGISTER_OPTIMIZER(Convolution, Adam);
  //FFNN_REGISTER_OPTIMIZER(Convolution, GradientDescent);

  /// Layer configuration parameters
  ReceptiveVolumeBankType receptors_;

  /**
   * @brief Weight optimization resource
   * @note  This will be the <code>optimizer::None</code> type by default
   * @see   setOptimizer
   */
  typename Optimizer::Ptr opt_;
};
}  // namespace layer
}  // namespace ffnn

#undef CONVOLUTION_OUTPUT_HEIGHT
#undef CONVOLUTION_OUTPUT_WIDTH
#undef CONVOLUTION_BASE_TARGS
#undef CONVOLUTION_VOLUME_TARGS
#undef CONVOLUTION_TARGS

/// FFNN (implementation)
//#include <ffnn/layer/impl/fully_connected.hpp>
#endif  // FFNN_LAYER_FULLY_CONNECTED_H
