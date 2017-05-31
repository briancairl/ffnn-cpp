/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_INTERNAL_SIGNATURE_H
#define FFNN_INTERNAL_SIGNATURE_H

// C++ Standard Library
#include <exception>
#include <typeinfo>
#include <typeindex>

// GCC
#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace ffnn
{
namespace internal
{
namespace signature
{
/**
 * @brief Generates a type signature string for a type
 * @return type signature
 * @note If using GCC, the signature will be human-readable.
 */
template<typename Type>
const std::string signature()
{
  const char* tag  = typeid(Type).name();
#ifdef __GNUC__
  int status;
  const char* type = abi::__cxa_demangle(tag, 0, 0, &status);
  return std::string(type);
#else
  return std::string(tag);
#endif
}

/**
 * @brief Apply type signature to an archive
 * @param[in,out] archive  output archive
 */
template<typename SerializableType, typename Archive>
void apply(Archive& archive)
{
  // Generate signature
  const std::string signature(signature<SerializableType>());

  // Write to archive
  archive & signature;
}

/**
 * @brief Checks a type signature read from an archive
 * @param[in,out] archive  input archive
 */
template<typename SerializableType, typename Archive>
void check(Archive& archive)
{
  // Generate signature
  const std::string expected(signature<SerializableType>());

  // Read signature from archive
  std::string signature;
  archive & signature;

  // Check type signature
  if (signature != expected)
  {
    FFNN_ERROR("Signature mismatch when unarchiving: " << signature << " != " << expected);
    throw std::runtime_error("Unarchiving signature mismatch.");
  }
}
}  // namespace signature
}  // namespace io
}  // namespace ffnn
#endif  // FFNN_IO_H
