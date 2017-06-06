/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
#define FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

namespace ffnn
{
namespace optimizer
{
template<typename LayerType,
         LossFunction LossFn = CrossEntropy>
class GradientDescent :
  public Optimizer<LayerType>
{
public:
  /// Base type standardization
  typedef Optimizer<LayerType> BaseType;

  /// Scalar type standardization
  typedef typename BaseType::Scalar Scalar;

  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::ParametersType ParametersType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(Scalar lr);
  virtual ~GradientDescent() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in,out] layer  Layer to optimize
   */
  void initialize(LayerType& layer);

  /**
   * @brief Resets persistent Optimizer states
   * @param[in,out] layer  Layer to optimize
   */
  void reset(LayerType& layer);

  /**
   * @brief Computes one forward optimization update step
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  bool forward(LayerType& layer);

  /**
   * @brief Computes optimization step during backward propogation
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  bool backward(LayerType& layer);

  /**
   * @brief Applies optimization update
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  bool update(LayerType& layer);

protected:
  /// Learning rate
  Scalar lr_;

  /// Previous input
  InputBlockType prev_input_;

  /// Coefficient gradient
  ParametersType gradient_;
};
}  // namespace optimizer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/optimizer/gradient_descent/gradient_descent.hpp>

/// FFNN (specializations)
//#include <ffnn/impl/optimizer/gradient_descent/convolution.hpp>
#include <ffnn/impl/optimizer/gradient_descent/fully_connected.hpp>
#endif  // FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
