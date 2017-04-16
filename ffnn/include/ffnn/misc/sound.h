/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_MISC_SOUND_H
#define FFNN_MISC_SOUND_H

// C++ Standard Library
#include <vector>

// libsndfile
#include <sndfile.h>

namespace ffnn
{
namespace misc
{
namespace sound
{
/**
 * @brief Reads sound file into multichannel array
 * @param filename  name of the file to read
 * @param[out] sound  object to store the loaded sound data
 */
template<typename ManagedArrayType>
SF_INFO read(const char* filename, ManagedArrayType& sound)
{
  using ValueType = typename ManagedArrayType::value_type;

  // Try to open sound filel get infor for file
  SF_INFO info;
  SNDFILE* file_handle = sf_open(filename, SFM_READ, &info);
  if (file_handle == NULL)
  {
    return SF_INFO();
  }

  // Read the sound
  sound.resize(info.frames * info.channels);
  if (info.frames != sf_readf_float(file_handle, const_cast<ValueType*>(sound.data()), info.frames))
  {
    return SF_INFO();
  }

  // Close the sound file
  sf_close(file_handle);

  // Compute output sampling rate
  return info;
}

template<typename ManagedArrayType>
bool write(const char* filename, const ManagedArrayType& sound, SF_INFO info)
{
  using ValueType = typename ManagedArrayType::value_type;

  // Open file handle
  SNDFILE* file_handle = sf_open(filename, SFM_WRITE, &info);
  if (file_handle == NULL)
  {
    return false;
  }

  // Write sound
  sf_write_float(file_handle, const_cast<ValueType*>(sound.data()), sound.size());

  // Close the sound file
  sf_close(file_handle);
  return true;
}

}  // namespace sound
}  // namespace misc
}  // namespace ffnn
#endif  // FFNN_MISC_SOUND_H