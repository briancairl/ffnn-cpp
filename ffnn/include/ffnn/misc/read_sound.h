/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_MISC_READ_SOUND_H
#define FFNN_MISC_READ_SOUND_H

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
 * @param downsample_factor  if > 1, outputted sound will be downsampled by (fs/downsample_factor)
 */
template<typename ValueType = float>
SF_INFO read(const char* filename, std::vector<std::vector<ValueType>>& sound, size_t downsample_factor = 1UL)
{
  // Try to open sound filel get infor for file
  SF_INFO info;
  SNDFILE* file_handle = sf_open(filename, SFM_READ, &info);
  if (file_handle == NULL)
  {
    return SF_INFO();
  }

  // Setup output vector
  sound.resize(info.channels);
  for (auto& channel : sound)
  {
    channel.reserve(info.frames);
  }

  // Read the sound
  ValueType* buffer = new ValueType[info.frames * info.channels];
  if (buffer != NULL && info.frames != sf_readf_float (file_handle, buffer, info.frames))
  {
    delete[] buffer;
    return SF_INFO();
  }

  // Close the sound file
  sf_close(file_handle);

  // Compute frame skip w/ specified resampling factor
  const size_t skip = (downsample_factor - 1UL) * info.channels;

  // Copy sound frames into channel buffers
  size_t chx = 0UL;
  size_t idx = 0UL;
  const size_t n_elements = static_cast<size_t>(info.frames) *
                            static_cast<size_t>(info.channels);
  while(idx < n_elements)
  {
    sound[chx++].push_back(buffer[idx++]);
  
    chx %= info.channels;
    idx += skip;
  }

  // Delete temporary buffer
  delete[] buffer;

  info.samplerate = static_cast<double>(info.samplerate) /
                    static_cast<double>(downsample_factor);

  // Compute output sampling rate
  return info;
}

template<typename ValueType = float>
bool write_single_channel(const char* filename, const std::vector<ValueType>& channel, SF_INFO info)
{
  // Open file handle
  info.channels = 1;
  SNDFILE* file_handle = sf_open(filename, SFM_WRITE, &info);
  if (file_handle == NULL)
  {
    return false;
  }

  // Write sound
  sf_writef_float(file_handle, channel.data(), channel.size());

  // Close the sound file
  sf_close(file_handle);
  return true;
}

}  // namespace sound
}  // namespace misc
}  // namespace ffnn
#endif  // FFNN_MISC_READ_SOUND_H