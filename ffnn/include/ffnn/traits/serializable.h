/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_TRAITS_SERIALIZABLE_H
#define FFNN_TRAITS_SERIALIZABLE_H

// C++ Standard Library
#include <iostream>

// Boost
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace traits
{
/**
 * @brief An object which can be saved and loaded
 */
class Serializable
{
public:
  /// Class-version type standardization
  typedef FFNN_CLASS_VERSION_TYPE ClassVersionType;

  /**
   * @brief Loads object contents
   * @param[in] is  input stream
   * @param version  class version number
   * @retval true  if object was loaded successfully
   * @retval false  otherwise
   */
  virtual bool load(std::istream& is, ClassVersionType version = 0) = 0; 

  /**
   * @brief Saves object contents
   * @param[out] os  input stream
   * @param version  class version number
   * @retval true  if object was loaded successfully
   * @retval false  otherwise
   */
  virtual bool save(std::ostream& os, ClassVersionType version = 0) = 0;

private:
  friend class boost::serialization::access;
};
}  // namespace traits
}  // namespace ffnn
#endif  // FFNN_TRAITS_SERIALIZABLE_H
