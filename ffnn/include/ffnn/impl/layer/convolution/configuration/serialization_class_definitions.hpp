/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 * @warning This file defines a block of code which directly inserted into a class definitions. Since these method are never to be
 *          called by a user and deal with an intrusive serialization implementation, they have been moved here to make some things
 *          look a bit nicer, as well to allow for quick toggling of serialziation support at compile-time
 */
#ifndef FFNN_IMPL_LAYER_CONVOLUTION_CONFIGURATION_SERIALIZATION_CLASS_DEFINITIONS_HPP
#define FFNN_IMPL_LAYER_CONVOLUTION_CONFIGURATION_SERIALIZATION_CLASS_DEFINITIONS_HPP
#pragma once

/// Allow boost serialization to call serialization methods of parent class
friend class boost::serialization::access;

/**
 * @brief Serializer
 * @param ar  input/output archive
 * @param version  archive versioning information
 */
template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
  ar & input_shape_;
  ar & output_shape_;
  ar & embedded_input_shape_;
  ar & embedded_output_shape_;
  ar & filter_shape_;
  ar & row_stride_;
  ar & col_stride_;
  ar & stride_shape_;
}
#endif  // FFNN_IMPL_LAYER_CONVOLUTION_CONFIGURATION_SERIALIZATION_CLASS_DEFINITIONS_HPP
