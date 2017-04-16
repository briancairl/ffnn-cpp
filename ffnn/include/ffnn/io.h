/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_TRAITS_SERIALIZABLE_H
#define FFNN_TRAITS_SERIALIZABLE_H

// C++ Standard Library
#include <iostream>

// FFNN
#include <ffnn/traits/serializable.h>

namespace ffnn
{
/**
 * @brief Saves a Serializable object
 * @param is input stream
 * @param object a Serializable object
 * @param version object version number
 */
inline void save(std::ostream& os,
                 const traits::Serializable& object,
                 const traits::Serializable::VersionType version = 0)
{
  traits::Serializable::OutputArchive archive(os);
  object.save(archive, version);
}

/**
 * @brief Loads a Serializable object
 * @param is input stream
 * @param object a Serializable object
 * @param version object version number
 */
inline void load(std::istream& is,
                 traits::Serializable& object,
                 const traits::Serializable::VersionType version = 0)
{
  // Read data
  traits::Serializable::InputArchive archive(is);
  object.load(archive, version);
}
}  // namespace ffnn
#endif  // FFNN_TRAITS_SERIALIZABLE_H
