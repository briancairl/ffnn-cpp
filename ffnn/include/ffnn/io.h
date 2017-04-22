/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_IO_H
#define FFNN_IO_H

// C++ Standard Library
#include <iostream>
#include <typeinfo>
#include <typeindex>

// FFNN
#include <ffnn/logging.h>

namespace ffnn
{
template<typename SerializableType>
inline void save(std::ostream& os,
                 const SerializableType& object,
                 const typename SerializableType::VersionType version = 0)
{
  typename SerializableType::OutputArchive archive(os);
  object.save(archive, version);
}

template<typename SerializableType>
inline void load(std::istream& is,
                 SerializableType& object,
                 const typename SerializableType::VersionType version = 0)
{
  // Read data
  typename SerializableType::InputArchive archive(is);
  object.load(archive, version);
}
}  // namespace ffnn
#endif  // FFNN_IO_H
