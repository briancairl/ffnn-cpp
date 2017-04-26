/**
 * @author Brian Cairl
 * @date 2017
 */

// C++ Standard Library
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

// FFNN
#include <ffnn/aligned_types.h>
#include <ffnn/logging.h>
#include <ffnn/layer/activation.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/activation.h>
#include <ffnn/layer/output.h>
#include <ffnn/neuron/lecun_sigmoid.h>
#include <ffnn/neuron/linear.h>
#include <ffnn/neuron/rectified_linear.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/io.h>

// Layer-type alias
using Layer  = ffnn::layer::Layer<float>;
using Input  = ffnn::layer::Input<float>;
using FC_H1  = ffnn::layer::FullyConnected<float>;
using ACT_1  = ffnn::layer::Activation<float, ffnn::neuron::Linear>;
using FC_H2  = ffnn::layer::FullyConnected<float>;
using ACT_2  = ffnn::layer::Activation<float, ffnn::neuron::RectifiedLinear>;
using Output = ffnn::layer::Output<float>;

void read_vector(std::ifstream& is, Eigen::VectorXf& v, size_t pad = 2)
{
  size_t n;
  is >> n;

  v.setZero(n + pad, 1);
  for (size_t idx = 0UL; idx < n; idx++)
  {
    is >> v(idx);
  }
}

// Run tests
int main(int argc, char** argv)
{
  FFNN_INFO("Reading inputs.");

  std::ifstream is(argv[1]);
  ffnn::aligned::Buffer<Eigen::VectorXf> samples;
  ffnn::aligned::Buffer<Eigen::VectorXf> targets;

  size_t count = 10000;
  while (!is.eof() && --count)
  {
    Eigen::VectorXf v;
    read_vector(is, v); targets.push_back(v);
    read_vector(is, v); samples.push_back(v);
  }

  // Layer sizes
  static const Layer::SizeType DIM = samples[0].rows();

  // Create layers
  auto input = boost::make_shared<Input>(DIM);  
  auto h1    = boost::make_shared<FC_H1>(DIM, FC_H1::Parameters(0.05));
  auto h2    = boost::make_shared<FC_H2>(DIM, FC_H2::Parameters(0.05));
  auto a2    = boost::make_shared<ACT_2>(ACT_2::Parameters(0.05));
  auto a1    = boost::make_shared<ACT_1>(ACT_1::Parameters(0.05));
  auto output = boost::make_shared<Output>();  

  // Create network
  std::vector<Layer::Ptr> layers({input, h1, a1, h2, a2, output});

  std::ifstream ifs("/home/brian/network_saves/990_5000.net", std::ios::binary);
  for(const auto& layer : layers)
  {
    try
    {
      ffnn::load(ifs, *layer);
    }
    catch (const boost::archive::archive_exception& ex)
    {
      FFNN_ERROR(ex.what());
    }
  }
  ifs.close();

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]);
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    if (!layer->initialize() || !layer->isInitialized())
    {
      FFNN_ERROR("Error initializing network.");
      return 1;
    }
  }

  std::ofstream ofs("/home/brian/network_saves/test_out.txt");
  // Check that error montonically decreases
  for (size_t idx = 0; samples.size(); idx++)
  {
    // Forward activate
    (*input) << samples[idx];
    for(const auto& layer : layers)
    {
      layer->forward();
    }
    Eigen::VectorXf netout(samples[idx].rows(), 1);
    (*output) >> netout;

    ofs << targets[idx].transpose() << std::endl;
    ofs << netout.transpose() << std::endl;
  }
  return 0;
}
