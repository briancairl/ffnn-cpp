/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_INTERNAL_TRAITS_UNIQUE_H
#define FFNN_INTERNAL_TRAITS_UNIQUE_H

// C++ Standard Library
#include <string>

// Boost
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace traits
{
/**
 * @brief An object with a unique ID
 */
class Unique :
  public Serializable
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
    setID(boost::uuids::to_string(id_gen_()));
  }

  /**
   * @brief Returns an ID
   * @return ID
   */
  inline const std::string& getID() const
  {
    return id_;
  }

  /**
   * @brief Sets an ID
   * @param id  ID to set
   */
  inline void setID(const std::string& id)
  {
    id_ = id;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE();

  void save(OutputArchive& ar, VersionType) const
  {
    ffnn::io::signature::apply<Unique>(ar);
    ar & id_;
  }

  void load(InputArchive& ar, VersionType)
  {
    ffnn::io::signature::check<Unique>(ar);
    ar & id_;
  }

private:
  /// UUID string associated with this class
  std::string id_;
};
}  // namespace traits
}  // namespace ffnn
#endif  // FFNN_INTERNAL_TRAITS_UNIQUE_H
