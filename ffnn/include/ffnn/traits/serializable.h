/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_TRAITS_SERIALIZABLE_H
#define FFNN_TRAITS_SERIALIZABLE_H

// C++ Standard Library
#include <iostream>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
template<typename Serializable>
void save(std::ostream& os,
          const Serializable& object,
          const typename Serializable::VersionType version = 0);

template<typename Serializable>
void load(std::istream& is,
          Serializable& object,
          const typename Serializable::VersionType version = 0);

namespace traits
{
/**
 * @brief An object which can be saved and loaded
 */
class Serializable
{
public:
  /// Class-version type standardization
  typedef FFNN_SERIALIZATION_VERSION_TYPE VersionType;

  /// Input (load) archive type
  typedef FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE InputArchive;

  /// Output (save) archive type
  typedef FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE OutputArchive;
  
  virtual void save(OutputArchive& ar, VersionType) const = 0;

  virtual void load(InputArchive& ar, VersionType) = 0;
};

/**
 * @brief Imbues an object with objects/types for serialization
 * @warning Must be placed in the <code>protected</code> portion of a class definition
 */
#define FFNN_REGISTER_SERIALIZABLE(object)\
  typedef ::ffnn::traits::Serializable::VersionType VersionType;\
  typedef ::ffnn::traits::Serializable::InputArchive InputArchive;\
  typedef ::ffnn::traits::Serializable::OutputArchive OutputArchive;\
  template<typename SerializableType>\
    friend void ::ffnn::save(std::ostream& os,\
                             const SerializableType& object,\
                             const typename SerializableType::VersionType version);\
  template<typename SerializableType>\
    friend void ::ffnn::load(std::istream& is,\
                             SerializableType& object,\
                             const typename SerializableType::VersionType version);

}  // namespace traits
}  // namespace ffnn
#endif  // FFNN_TRAITS_SERIALIZABLE_H
