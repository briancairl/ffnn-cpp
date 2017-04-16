/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_TRAITS_UNIQUE_H
#define FFNN_TRAITS_UNIQUE_H

// C++ Standard Library
#include <string>

// Boost
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace ffnn
{
namespace traits
{
/**
 * @brief An object with a unique ID
 */
class Unique
{
public:
  /**
   * @brief Default constructor
   */
  Unique() 
  {
    // UUID generator
    static boost::uuids::basic_random_generator<boost::mt19937> id_gen_;

    // Generate a new UUID
    this->id_ = boost::uuids::to_string(id_gen_());
  }

  /**
   * @brief Returns a unique ID associated a class
   * @return unique ID
   */
  inline const std::string& id() const { return id_; }

protected:
  /// UUID string associated with this class
  std::string id_;
};
}  // namespace traits
}  // namespace ffnn
#endif  // FFNN_TRAITS_UNIQUE_H
