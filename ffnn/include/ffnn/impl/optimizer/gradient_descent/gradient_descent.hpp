/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 */
#ifndef FFNN_IMPL_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
#define FFNN_IMPL_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP

// C++ Standard Library
#include <exception>

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/optimizer/none.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace optimizer
{
template<typename LayerType,
         LossFunction LossFn>
GradientDescent_<LayerType, LossFn>::GradientDescent_(Scalar lr) :
  BaseType("GradientDescent"),
  lr_(lr)
{}

template<typename LayerType,
         LossFunction LossFn>
void GradientDescent_<LayerType, LossFn>::initialize(LayerType& layer)
{
  FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

  // Capture gradient sizing
  gradient_ = layer.parameters_;

  // Capture input sizing
  prev_input_ = layer.input_;

  // Reset optimizer
  reset(layer);
}

template<typename LayerType,
         LossFunction LossFn>
void GradientDescent_<LayerType, LossFn>::reset(LayerType& layer)
{
  FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

  // Reset gradient
  gradient_.setZero();
}

template<typename LayerType,
         LossFunction LossFn>
bool GradientDescent_<LayerType, LossFn>::forward(LayerType& layer)
{
  FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

  // Copy current input for updating
  prev_input_.noalias() = layer.input_;
  return true;
}

template<typename LayerType,
         LossFunction LossFn>
bool GradientDescent_<LayerType, LossFn>::update(LayerType& layer)
{
  FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

  // Incorporate learning rate
  gradient_ *= lr_;

  // Update parameters
  layer.parameters_ -= gradient_;

  // Reinitialize optimizer
  reset(layer);
  return true;
}
}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_IMPL_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
