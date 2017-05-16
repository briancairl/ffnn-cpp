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
#include <ffnn/optimizer/fwd.h>

/// FFNN (specializations)
#include <ffnn/optimizer/impl/gradient_descent/fully_connected.hpp>
#include <ffnn/optimizer/impl/gradient_descent/local_convolution.hpp>
#include <ffnn/optimizer/impl/gradient_descent/sparsely_connected.hpp>

#endif  // FFNN_LAYER_OPTIMIZATION_GRADIENT_DESCENT_H
