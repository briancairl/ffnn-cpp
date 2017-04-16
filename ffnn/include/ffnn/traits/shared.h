/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_TRAITS_SHARED_H
#define FFNN_TRAITS_SHARED_H

// Boost
#include <boost/shared_ptr.hpp>

namespace ffnn
{
namespace traits
{
/**
 * @brief Imbues an object with shared resource pointer types
 */
template<typename Object>
class Shared
{
public:
  /// Shared resource standardization
  typedef boost::shared_ptr<Object> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const Object> ConstPtr;
};
}  // namespace traits
}  // namespace ffnn
#endif  // FFNN_TRAITS_SHARED_H
