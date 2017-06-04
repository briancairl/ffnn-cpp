/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_CONVOLUTION_H
#define FFNN_LAYER_CONVOLUTION_H

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>

#include <ffnn/distribution/distribution.h>
#include <ffnn/distribution/normal.h>

#include <ffnn/layer/layer.h>
#include <ffnn/layer/hidden.h>

#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/optimizer.h>

#include <ffnn/layer/convolution/filter.h>
#include <ffnn/layer/convolution/sizing.h>
#include <ffnn/layer/convolution/configuration.h>
#include <ffnn/layer/convolution/compile_time_options.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief A convolution layer
 */
template <typename ValueType,
          typename Options    = typename convolution::options<>,
          typename Extrinsics = typename convolution::extrinsics<ValueType, Options>>
class Convolution :
  public Extrinsics::HiddenLayerType
{
  FFNN_ASSERT_DONT_MODIFY_EXTRINSICS(convolution);
public:
  /// Self type alias
  using SelfType = Convolution<ValueType, Options, Extrinsics>;

  /// Base type alias
  using BaseType = typename Extrinsics::HiddenLayerType;

  /// Configuration type standardization
  typedef convolution::Configuration<SelfType, ValueType, Options, Extrinsics> Configuration;

  /// Dimension type standardization
  typedef typename BaseType::ShapeType ShapeType;

  /// Filter parameters type standardization
  typedef typename Extrinsics::ParametersType ParametersType;

  /// 2D-value mapping standardization
  typedef typename Extrinsics::ForwardMappingGridType ForwardMappingGridType;

  /**
   * @brief Setup constructor
   * @param config  Layer configuration struct
   */
  explicit Convolution(const Configuration& config = Configuration());
  virtual ~Convolution();

  /**
   * @brief Initialize the layer
   * @retval true  if layer was initialized successfully
   * @retval false otherwise
   *
   * @warning If layer is not loaded instance, this method will initialize layer sizings
   *          but weights and biases will be zero
   */
  bool initialize();

  /**
   * @brief Performs forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  bool forward();

  /**
   * @brief Performs backward error propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   * @warning Does not apply layer weight updates
   * @warning Will throw if an optimizer has not been associated with this layer
   */
  bool backward();

  /**
   * @brief Applies accumulated layer weight updates computed during optimization
   * @retval true  if weight update succeeded
   * @retval false  otherwise
   * @warning Will throw if an optimizer has not been associated with this layer
   */
  bool update();

  /**
   * @brief Exposes layer parameters
   * @return Convolution filter parameters
   */
  inline const ParametersType& getParameters() const
  {
    return parameters_;
  }

private:
  //FFNN_REGISTER_OPTIMIZER(Convolution, Adam);
  FFNN_REGISTER_OPTIMIZER(Convolution, GradientDescent);

  /**
   * @brief Reset all internal volumes
   */
  void reset();

  /// Layer configurations
  Configuration config_;

  /**
   * @brief Layer parameters
   * @note  For the Convolution layer, theses are Filter coefficients
   */
  ParametersType parameters_;

  /// Forward error mapping grid
  ForwardMappingGridType forward_error_mappings_;

  /// Output value mapping grid
  ForwardMappingGridType output_mappings_;

  /**
   * @brief Maps outputs of this layer to inputs of the next
   * @param next  a subsequent layer
   * @param offset  offset index of a memory location in the input buffer of the next layer
   * @retval <code>offset + output_shape_.size()</code>
   */
  offset_type connectToForwardLayer(const Layer<ValueType>& next, offset_type offset);

#ifndef FFNN_NO_SERIALIZATION_SUPPORT
protected:
  FFNN_REGISTER_SERIALIZABLE(Convolution);

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const;

  /// Load serializer
  void load(InputArchive& ar, VersionType version);
#endif
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/layer/convolution.hpp>
#endif  // FFNN_LAYER_CONVOLUTION_H
