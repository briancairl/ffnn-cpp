/** 
 * @mainpage
 *
 * @section introduction Introduction
 * @{
 *    FFNN is a C++ header-only neural network library. The library makes heavy use of
 *    the Eigen matrix library to take advantage of compile-time matrix optimizations
 *    and hardware vectorization.
 * @}
 *
 * @section config Compile-time configurations
 * @{
 * - <code>FFNN_ARCHIVE_TEXT</code>: forces save data to be stored as human-readable test
 * - <code>FFNN_NO_OPTIMIZER_SUPPORT</code>: force-excluded network training support from compilation
 * - <code>FFNN_NO_ASSERT</code>: disables run-time assertions
 * - <code>FFNN_NO_LOGGING</code>: disables debugging printouts
 * @}
 *
 * @section overrides Compile-time configuration overrides
 * @{
 * - <code>FFNN_ALLOW_ASSERT</code>: force enables run-time assertions
 * - <code>FFNN_ALLOW_LOGGING</code>: force enables debugging printouts
 * - <code>FFNN_SUPRESS_ERROR_LOGGING</code>: suppresses error message
 * - <code>FFNN_NO_EXPLICIT_ALIGNMENT</code>: disables matrix vectorization
 * @}
 * 
 * @author Brian Cairl
 * @date April 2017
 */
#ifndef FFNN_GLOBAL_H
#define FFNN_GLOBAL_H

// C++ Standard Library
#include <cstddef>

// Global Types
#ifndef FFNN_SIZE_TYPE
#define FFNN_SIZE_TYPE int32_t
#endif
#ifndef FFNN_OFFSET_TYPE
#define FFNN_OFFSET_TYPE std::ptrdiff_t
#endif

// Serialization
#define FFNN_SERIALIZATION_VERSION_TYPE const unsigned int
#ifdef FFNN_ARCHIVE_TEXT
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#define FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE boost::archive::text_oarchive
#define FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE boost::archive::text_iarchive
#else
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#define FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE boost::archive::binary_oarchive
#define FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE boost::archive::binary_iarchive
#endif

// Eigen (addons)
#include <boost/serialization/array.hpp>
#define EIGEN_DENSEBASE_PLUGIN "ffnn/config/eigen/dense_base_addons.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif  // FFNN_GLOBAL_H
