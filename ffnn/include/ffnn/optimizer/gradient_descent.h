/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
#define FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace optimizer
{
template<typename LayerType>
class GradientDescent :
  public Optimizer<LayerType>
{
public:
  /**
   * @brief Setup constructor
   */
  GradientDescent() :
    Optimizer<LayerType>("GradientDescent")
  {}
  virtual ~GradientDescent()
  {}
};
}  // namespace optimizer
}  // namespace ffnn

/// FFNN (specializations)
#include <ffnn/optimizer/impl/fully_connected/gradient_descent.hpp>
#include <ffnn/optimizer/impl/sparsely_connected/gradient_descent.hpp>
#endif  // FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
