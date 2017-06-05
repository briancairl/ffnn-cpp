/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
#define FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H

// FFNN
#include <ffnn/internal/config.h>
#include <ffnn/assert.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>

template<typename ValueType,
         typename LayerType>
class GradientDescent :
  public Optimizer<LayerType>
{
public:
  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::ParametersType ParametersType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ValueType lr);
  virtual ~GradientDescent() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  void initialize(LayerType& layer);

  /**
   * @brief Resets persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  void reset(LayerType& layer);

  /**
   * @brief Computes one forward optimization update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  bool forward(LayerType& layer);

  /**
   * @brief Computes optimization step during backward propogation
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  bool backward(LayerType& layer);

  /**
   * @brief Applies optimization update
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  bool update(LayerType& layer);

protected:
  /// Learning rate
  ValueType lr_;

  /// Previous input
  InputBlockType prev_input_;

  /// Coefficient gradient
  ParametersType gradient_;
};
}  // namespace optimizer
}  // namespace ffnn

/// FFNN (specializations)
#include <ffnn/optimizer/impl/gradient_descent/convolution.hpp>
#include <ffnn/optimizer/impl/gradient_descent/fully_connected.hpp>

#endif  // FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
