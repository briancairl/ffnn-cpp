/**
 * @author Brian Cairl
 * @date 2017
 * @brief Reference: https://arxiv.org/abs/1412.6980
 */
#ifndef FFNN_LAYER_OPTIMIZATION_ADAM_H
#define FFNN_LAYER_OPTIMIZATION_ADAM_H

// C++ Standard Library
#include <type_traits>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/internal/config.h>
#include <ffnn/optimizer/optimizer.h>
#include <ffnn/optimizer/fwd.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/impl/optimizer/adam/adam_states.hpp>

namespace ffnn
{
namespace optimizer
{
template<typename LayerType,
         LossFunction LossFn = CrossEntropy>
class Adam :
  public Optimizer<LayerType>
{
public:
  /// Base type standardization
  typedef Optimizer<LayerType> BaseType;

  /// Scalar type standardization
  typedef typename BaseType::Scalar Scalar;

  /// Matrix type standardization
  typedef typename LayerType::ParametersType ParametersType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  Adam(Scalar lr, Scalar beta1 = 0.9, Scalar beta2 = 0.999, Scalar eps = 1e-3) :
    BaseType("Adam"),
    states_(beta1, beta2, eps),
    gd_(lr)
  {}
  virtual ~Adam() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in,out] layer  Layer to optimize
   */
  inline void initialize(LayerType& layer)
  {
    // Initialize gradient states
    states_.initialize(layer);

    // Initialize base optimizer
    gd_.initialize(layer);
  }

  /**
   * @brief Resets persistent Optimizer states
   * @param[in,out] layer  Layer to optimize
   */
  inline void reset(LayerType& layer)
  {
    gd_.reset(layer);
  }

  /**
   * @brief Computes one forward optimization update step
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  inline bool forward(LayerType& layer)
  {
    return gd_.forward(layer);
  }

  /**
   * @brief Computes optimization step during backward propogation
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization setup was successful
   * @retval false  otherwise
   */
  inline bool backward(LayerType& layer)
  {
    return gd_.backward(layer);
  }

  /**
   * @brief Applies optimization update
   * @param[in,out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  inline bool update(LayerType& layer)
  {
    // Update gradient states
    states_.update(gd_.gradient_);

    // Update base optimizer
    return gd_.update(layer);
  }

protected:
  /// Holds running mean/variance estimates
  AdamStates<LayerType> states_;

  /// Coefficient gradient
  GradientDescent<LayerType, LossFn> gd_;
};
}  // namespace optimizer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/impl/optimizer/gradient_descent/gradient_descent.hpp>
#endif  // FFNN_LAYER_OPTIMIZATION_ADAM_H
