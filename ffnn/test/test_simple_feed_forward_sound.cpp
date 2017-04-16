/**
 * @author Brian Cairl
 * @date 2017
 */

// C++ Standard Library
#include <cmath>
#include <exception>
#include <iostream>
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

#include <ffnn/neuron/leaky_rectified_linear.h>
#include <ffnn/neuron/rectified_linear.h>
#include <ffnn/neuron/linear.h>
#include <ffnn/neuron/sigmoid.h>

// FFNN Miscellany 
#include <ffnn/misc/read_sound.h>


using Layer  = ffnn::layer::Layer<float>;
using Input  = ffnn::layer::Input<float, 50>;
using Layer1 = ffnn::layer::FullyConnected<float, ffnn::neuron::LeakyRectifiedLinear>;
using Layer2 = ffnn::layer::FullyConnected<float, ffnn::neuron::Linear>;
using Output = ffnn::layer::Output<float, 50>;

using MappedVector = ffnn::Mapped<Eigen::VectorXf>;

MappedVector::Ptr chunk(float* raw, Layer::SizeType idx, Layer::SizeType length)
{
  return MappedVector::create(raw + idx, length, 1UL);
}

TEST(TestFullyConnected, FullyConnectedForward)
{
  std::vector<std::vector<float>> sound;
  const auto info = ffnn::misc::sound::read("/home/brian/packages/src/ffnn/ffnn/test/sample_short.wav", sound, 1);

  static const size_t ITERATIONS = 1000;
  static const size_t CHUNK_SIZE = 50;
  static const size_t CHUNK_STEP = 15;

  static const Layer::SizeType DIM_0 = CHUNK_SIZE;
  static const Layer::SizeType DIM_1 = 30;
  static const Layer::SizeType DIM_2 = CHUNK_SIZE;

  // Create two layers to connect
  auto input_layer = boost::make_shared<Input>(DIM_0);
  auto output_layer = boost::make_shared<Output>();

  std::vector<Layer::Ptr> layers(4UL);

  // Allocate layers
  layers[0] = input_layer;
  {
    Layer1::Config config_;
    config_.learning_rate = 10.0 / static_cast<float>(sound[0].size());
    config_.weight_init_variance = 1e-0/std::sqrt(2.0);
    layers[1] = boost::make_shared<Layer1>(DIM_1, config_);
  }
  {
    Layer2::Config config_;
    config_.learning_rate = 10.0 / static_cast<float>(sound[0].size());
    config_.weight_init_variance = 1e-0/std::sqrt(2.0);
    layers[2] = boost::make_shared<Layer2>(DIM_2, config_);
  }
  layers[3] = output_layer;

  // Connect layers
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[0], layers[1])); // input-->net
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[1], layers[2]));
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[1])); // recurrence
  } 
  EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[2], layers[3])); // net-->output

  // Initialize and check all layers
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  // Pretend training on constant target
  Eigen::VectorXf output(CHUNK_SIZE, 1);

  std::vector<float> sound_out(sound[0].size());
  MappedVector::Type mapped_sound_out(sound_out.data(), sound_out.size(), 1);

  std::vector<float> sound_weights(sound[0].size());
  MappedVector::Type mapped_sound_weights(sound_weights.data(), sound_weights.size(), 1);

  float MSE = 0.0;
  for (size_t itr = 0UL; itr < ITERATIONS; itr++)
  {
    sound_out.resize(sound[0].size(), 0.0);

    size_t jdx = 0UL;
    for (size_t idx = 0UL; idx < sound[0].size(); idx+=CHUNK_STEP, jdx++)
    {
      if (idx + CHUNK_SIZE > sound[0].size())
      {
        break;
      }

      MappedVector::Ptr encoding_target = chunk(sound[0].data(), idx, CHUNK_SIZE);

      // Forward-propagate on all layers
      (*input_layer) << *encoding_target;
      for(const auto& layer : layers)
      {
        EXPECT_TRUE(layer->forward());
      }
      (*output_layer) >> output;

      // Compute output error and backward-propagate on all layers
      (*output_layer) << *encoding_target;
      for(const auto& layer : layers)
      {
        EXPECT_TRUE(layer->backward());
      }

      MSE += (output - *encoding_target).squaredNorm();

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
    //mapped_sound_out.array() /= mapped_sound_weights.array();
    float center = mapped_sound_out.mean();
    float range  = (mapped_sound_out.maxCoeff() - mapped_sound_out.minCoeff()) / 2.0;
    mapped_sound_out.array() -= center;
    mapped_sound_out.array() /= range;

    // Write out sound
    std::stringstream out_filename;
    out_filename << "/home/brian/packages/src/ffnn/ffnn/test/output/sample_short_" << itr << ".wav";
    std::string name = out_filename.str();

    FFNN_WARN_NAMED("sndfile", name);
    if (ffnn::misc::sound::write_single_channel(name.c_str(), sound_out, info))
    {
      FFNN_DEBUG_NAMED("sndfile", name);
    }
  }
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}