/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warning Do not include directly
 * @warning This file defines a block of code which directly inserted into a class definitions. Since these method are never to be
 *          called by a user and deal with an intrusive serialization implementation, they have been moved here to make some things
 *          look a bit nicer, as well to allow for quick toggling of serialziation support at compile-time
 */
#ifndef FFNN_IMPL_LAYER_CONVOLUTION_FILTER_SERIALIZATION_CLASS_DEFINITIONS_HPP
#define FFNN_IMPL_LAYER_CONVOLUTION_FILTER_SERIALIZATION_CLASS_DEFINITIONS_HPP
#pragma once

/// Allow boost serialization to call serialization methods of parent class
friend class boost::serialization::access;

/**
 * @brief Save serializer
 * @param ar  output archive
 * @param version  archive versioning information
 */
template<class Archive>
void save(Archive & ar, const unsigned int version) const
{
  size_type n = this->size();

  ar & bias;
  ar & n;
  for (const auto& kernel : *this)
  {
    ar & kernel;
  }
}

/**
 * @brief Load serializer
 * @param ar  input archive
 * @param version  archive versioning information
 * @note Statically sized version
 */
template<class Archive, bool T = Options::has_fixed_kernel_count>
typename std::enable_if<T>::type
  load(Archive & ar, const unsigned int version)
{
  size_type n = this->size();

  ar & bias;
  ar & n ;
  for (size_type idx = 0; idx < n; idx++)
  {
    ar & (*this)[idx];
  }
}
template<class Archive, bool T = Options::has_fixed_kernel_count>
typename std::enable_if<!T>::type
  load(Archive & ar, const unsigned int version)
{
  size_type n;

  ar & bias;
  ar & n;

  this->resize(n);
  for (size_type idx = 0; idx < n; idx++)
  {
    ar & (*this)[idx];
  }
}

/**
 * @brief Serializer
 * @param ar  input/output archive
 * @param version  archive versioning information
 */
template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
  boost::serialization::split_member(ar, *this, file_version);
}
#endif  // FFNN_IMPL_LAYER_CONVOLUTION_FILTER_SERIALIZATION_CLASS_DEFINITIONS_HPP
