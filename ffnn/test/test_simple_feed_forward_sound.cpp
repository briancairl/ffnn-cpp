/**
 * @author Brian Cairl
 * @date 2017
 */

// C++ Standard Library
#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// GTest
#include <gtest/gtest.h>

// FFNN
#undef FFNN_NO_LOGGING
#include <ffnn/mapped.h>

#include <ffnn/layer/layer.h>
#include <ffnn/layer/fully_connected.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/output.h>
#include <ffnn/layer/dropout.h>

#include <ffnn/neuron/leaky_rectified_linear.h>
#include <ffnn/neuron/rectified_linear.h>
#include <ffnn/neuron/linear.h>
#include <ffnn/neuron/sigmoid.h>

// FFNN Miscellany 
#include <ffnn/misc/sound.h>


using Layer  = ffnn::layer::Layer<float>;
using Input  = ffnn::layer::Input<float>;
using Layer1 = ffnn::layer::FullyConnected<float, ffnn::neuron::LeakyRectifiedLinear>;

using Layer2 = ffnn::layer::Dropout<float>;
using Layer3 = ffnn::layer::FullyConnected<float, ffnn::neuron::Sigmoid>;
using Output = ffnn::layer::Output<float>;
using MappedVector = ffnn::Mapped<Eigen::VectorXf>;

MappedVector::Type chunk(float* raw, Layer::SizeType idx, Layer::SizeType length)
{
  return MappedVector::Type(raw + idx, length, 1UL);
}

template<typename InputVector, typename OutputVector>
void diff(const InputVector src, OutputVector& dst, size_t smooth_order = 1)
{
  using Real = typename InputVector::Scalar;
  using Index = typename InputVector::Index;

  dst.setZero(src.rows(), src.cols());
  for (Index idx = 0; idx < src.size() - smooth_order - 1; idx++)
  {
    Real sum = 0;
    for (Index jdx = 0; jdx < smooth_order; jdx+=2)
    {
      sum += src(idx + jdx) - src(idx);
    }
    dst(idx) = sum / static_cast<Real>(smooth_order);
  } 
}


TEST(TestFullyConnected, FullyConnectedForward)
{
  Layer::BufferType sound;
  const auto info = ffnn::misc::sound::read("/home/brian/Music/Music/Samples/songs/test.wav", sound);

  static const size_t ITERATIONS = 1000;
  static const size_t CHUNK_SIZE = 64;
  static const size_t CHUNK_STEP = 16;

  static const Layer::SizeType DIM_0 = CHUNK_SIZE;
  static const Layer::SizeType DIM_1 = 64;
  static const Layer::SizeType DIM_2 = CHUNK_SIZE;

  // Create two layers to connect
  auto input_layer  = boost::make_shared<Input>(DIM_0);
  auto output_layer = boost::make_shared<Output>();

  std::vector<Layer::Ptr> layers(5UL);

  // Fully connected
  layers[0] = input_layer;
  {
    float wi = 1e-0f/std::sqrt(2.0);
    auto layer = boost::make_shared<Layer1>(DIM_1, Layer1::Parameters(wi, wi));
    auto lr = 5.0f / static_cast<float>(sound.size());
    layer->setOptimizer(boost::make_shared<Layer1::Opt::FirstOrderGradientDescent>(lr));
    layers[1] = layer;
  }

  // Dropout
  layers[2] = boost::make_shared<Layer2>(Layer2::Parameters(0.1));

  // Fully connected
  {
    float wi = 1e-0f/std::sqrt(2.0);
    auto layer = boost::make_shared<Layer3>(DIM_2, Layer3::Parameters(wi, wi));
    auto lr = 1.0f / static_cast<float>(sound.size());
    layer->setOptimizer(boost::make_shared<Layer3::Opt::FirstOrderGradientDescent>(lr));
    layers[3] = layer;
  }
  layers[4] = output_layer;

  // Connect layers
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[0], layers[1])); // input-->net
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[1], layers[2]));
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[3]));
  } 
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[3], layers[4])); // net-->output

  // Initialize and check all layers
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Pretend training on constant target
  Eigen::VectorXf output(CHUNK_SIZE, 1);


  MappedVector::Type mapped_sound(sound.data(), sound.size(), 1);
  Eigen::VectorXf sound_diff;
  diff(mapped_sound, sound_diff, 200);

  Layer::BufferType sound_out(sound.size(), 0);
  MappedVector::Type mapped_sound_out(sound_out.data(), sound_out.size(), 1);

  Layer::BufferType sound_weights(sound.size(), 0);
  MappedVector::Type mapped_sound_weights(sound_weights.data(), sound_weights.size(), 1);

  float MSE = 0.0;
  for (size_t itr = 0UL; itr < ITERATIONS; itr++)
  {
    size_t jdx = 0UL;
    for (size_t idx = 0UL; idx < sound.size(); idx+=CHUNK_STEP, jdx++)
    {
      if (idx + CHUNK_SIZE > sound.size())
      {
        break;
      }

      MappedVector::Type encoding_input = chunk(sound_diff.data(), idx, CHUNK_SIZE);
      MappedVector::Type encoding_target = chunk(sound.data(), idx, CHUNK_SIZE);

      // Forward-propagate on all layers
      (*input_layer) << encoding_input;
      for(const auto& layer : layers)
      {
        EXPECT_TRUE(layer->forward());
      }
      (*output_layer) >> output;

      // Compute output error and backward-propagate on all layers
      (*output_layer) << encoding_target;
      for(const auto& layer : layers)
      {
        EXPECT_TRUE(layer->backward());
      }

      MSE += (output - encoding_target).squaredNorm();

      MappedVector::Type(sound_out.data() + idx, CHUNK_SIZE, 1) += output;
      MappedVector::Type(sound_weights.data() + idx, CHUNK_SIZE, 1).array() += 1.0;
    }
    FFNN_DEBUG_NAMED("layer::FullyConnected", itr <<  " of "  << ITERATIONS << " with MSE: " << MSE/static_cast<float>(jdx));

    // Apply weights to all layers
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->update());
    }
    MSE = 0.0;

    // Normalize sound
    {
      float center = mapped_sound_out.mean();
      float range  = (mapped_sound_out.maxCoeff() - mapped_sound_out.minCoeff()) / 2.0;
      mapped_sound_out.array() -= center;
      mapped_sound_out.array() /= range;
    }

    // Write out sound
    std::stringstream out_filename;
    out_filename << "/home/brian/Music/Music/Samples/songs/output/sample_short_" << itr << ".wav";
    std::string name = out_filename.str();

    FFNN_WARN_NAMED("sndfile", name);
    if (ffnn::misc::sound::write(name.c_str(), sound_out, info))
    {
      FFNN_DEBUG_NAMED("sndfile", name);
    }
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////


  Layer::BufferType test_sound;
  const auto test_info = ffnn::misc::sound::read("/home/brian/Music/Music/Samples/songs/test_test.wav", test_sound);
  Eigen::VectorXf test_sound_diff;

  MappedVector::Type mapped_test_sound(test_sound.data(), test_sound.size(), 1);
  diff(mapped_test_sound, test_sound_diff, 200);

  Layer::BufferType test_sound_out(test_sound.size(), 0);
  MappedVector::Type mapped_test_sound_out(test_sound_out.data(), test_sound_out.size(), 1);

  for (size_t idx = 0UL; idx < test_sound.size() - CHUNK_SIZE; idx+=CHUNK_STEP)
  {
    MappedVector::Type encoding_input = chunk(test_sound_diff.data(), idx, CHUNK_SIZE);

    // Forward-propagate on all layers
    (*input_layer) << encoding_input;
    for(const auto& layer : layers)
    {
      EXPECT_TRUE(layer->forward());
    }
    (*output_layer) >> output;

    MappedVector::Type(test_sound_out.data() + idx, CHUNK_SIZE, 1) += output;
  }

  // Normalize sound
  {
    float center = mapped_test_sound_out.mean();
    float range  = (mapped_test_sound_out.maxCoeff() - mapped_test_sound_out.minCoeff()) / 2.0;
    mapped_test_sound_out.array() -= center;
    mapped_test_sound_out.array() /= range;
  }

  // Write out sound
  std::stringstream out_filename;
  out_filename << "/home/brian/Music/Music/Samples/songs/output/test_sample_short.wav";
  std::string name = out_filename.str();

  FFNN_WARN_NAMED("sndfile", name);
  if (ffnn::misc::sound::write(name.c_str(), test_sound_out, test_info))
  {
    FFNN_DEBUG_NAMED("sndfile", name);
  }
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}