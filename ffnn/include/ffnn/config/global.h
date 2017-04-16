
/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_GLOBAL_H
#define FFNN_GLOBAL_H

// Global Types
#define FFNN_SIZE_TYPE int32_t
#define FFNN_OFFSET_TYPE size_t
#define FFNN_CLASS_VERSION_TYPE const unsigned int

// Eigen (addons)
#include <boost/serialization/array.hpp>
#define EIGEN_DENSEBASE_PLUGIN "ffnn/config/eigen/eigen_dense_base_addons.h"
#include <Eigen/Core>
#endif  // FFNN_GLOBAL_H
