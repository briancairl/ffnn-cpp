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

/**
 * @brief Checks if a type fufills the requirements of the Distribution concept
 * @param Object object to check
 */
template<typename Object>
struct is_distribution
{
private:
  typedef typename Object::Scalar ValueType;

  template<typename U>
  static auto test_generate(int) -> decltype(std::declval<U>().generate(), std::true_type());
 
  template<typename U>
  static auto test_cdf(int, ValueType v) -> decltype(std::declval<U>().cdf(v), std::true_type());

  template<typename>
  static std::false_type test_generate(...);

  template<typename>
  static std::false_type test_cdf(...);

public:
  constexpr static bool value =
    std::integral_constant<
      bool,
      std::is_same<decltype(test_generate<Object>(0)), std::true_type>::value &&
      std::is_same<decltype(test_cdf<Object>(0, ValueType(0))), std::true_type>::value
    >::value;
};
}  // namespace traits
}  // namespace internal
}  // namespace ffnn
#endif  // FFNN_INTERNAL_TRAITS_H
