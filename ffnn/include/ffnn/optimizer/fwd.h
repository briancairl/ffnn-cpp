/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_FWD_H
#define FFNN_LAYER_OPTIMIZATION_FWD_H

// FFNN
#include <ffnn/optimizer/loss_function.h>

namespace ffnn
{
namespace optimizer
{
// Foward Declarations
template<typename LayerType> class None;
template<typename LayerType, LossFunction LossFn> class Adam;
template<typename LayerType, LossFunction LossFn> class Adam_;
template<typename LayerType, LossFunction LossFn> class GradientDescent;
template<typename LayerType, LossFunction LossFn> class GradientDescent_;

}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H
