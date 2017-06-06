/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_LOSS_FUNCTION_H
#define FFNN_LAYER_OPTIMIZATION_LOSS_FUNCTION_H

namespace ffnn
{
namespace optimizer
{
/**
 * @brief Loss-function enumerations
 */
typedef enum
{
  CrossEntropy,
  L2,
  L1,
  Hinge
}
LossFunction;

}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZATION_LOSS_FUNCTION_H
