/**
 * @author Brian Cairl
 * @date 2017
 * @brief Reference: https://arxiv.org/abs/1412.6980
 */
#ifndef FFNN_LAYER_OPTIMIZATION_ADAM_H
#define FFNN_LAYER_OPTIMIZATION_ADAM_H

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/optimizer/fwd.h>

/// FFNN (specializations)
#include <ffnn/optimizer/impl/adam/activation.hpp>
#include <ffnn/optimizer/impl/adam/fully_connected.hpp>
#include <ffnn/optimizer/impl/adam/sparsely_connected.hpp>

#endif  // FFNN_LAYER_OPTIMIZATION_ADAM_H
