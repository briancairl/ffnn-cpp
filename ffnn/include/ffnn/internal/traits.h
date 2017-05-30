/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_INTERNAL_TRAITS_H
#define FFNN_INTERNAL_TRAITS_H

// C++ Standard Library
#include <type_traits>

namespace ffnn
{
namespace internal
{
namespace traits
{
/**
 * @brief Has <code>value_type == std::true_type</code> if <code>Object</code> is a
 *        16-byte alignable type.
 *
 *        Has <code>value_type == std::false_type</code> otherwise
 * @param Object object to check
 */
template<class Object>
struct is_alignable_128 :
  std::conditional<
    (sizeof(Object)%16) == 0,
    std::true_type,
    std::false_type
  >::type
{};
}  // namespace traits
}  // namespace internal
}  // namespace ffnn
#endif  // FFNN_INTERNAL_TRAITS_H
