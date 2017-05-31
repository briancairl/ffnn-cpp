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

namespace ffnn
{
namespace io
{
namespace signature
{
/**
 * @brief Creates a type-signature for a SerializableType
 * @return type signature string
 */
template<typename SerializableType>
const char* signature()
{
  return typeid(SerializableType).name();
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
