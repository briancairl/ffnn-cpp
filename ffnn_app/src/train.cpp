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
#include <ffnn/neuron/sigmoid.h>
#include <ffnn/neuron/linear.h>
#include <ffnn/neuron/leaky_rectified_linear.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/io.h>

template<typename ValueType>
class Leaky :
  public ffnn::neuron::LeakyRectifiedLinear<ValueType, 1, 100>
{};

// Layer-type alias
using Layer  = ffnn::layer::Layer<float>;
using Input  = ffnn::layer::Input<float>;
using FC_H1  = ffnn::layer::FullyConnected<float>;
using FC_H2  = ffnn::layer::FullyConnected<float>;
using ACT_1  = ffnn::layer::Activation<float, ffnn::neuron::Linear>;
using ACT_2  = ffnn::layer::Activation<float, Leaky>;
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

  const size_t iterations = 10000;
  const size_t epoch = 10000-1;

  // Layer sizes
  static const Layer::SizeType DIM = samples[0].rows();

  // Create layers
  auto input = boost::make_shared<Input>(DIM);  
  auto h1    = boost::make_shared<FC_H1>(DIM, FC_H1::Parameters(1.0/(DIM*DIM)));
  auto h2    = boost::make_shared<FC_H2>(DIM, FC_H2::Parameters(1.0/(DIM*DIM)));
  auto a1    = boost::make_shared<ACT_1>();
  auto a2    = boost::make_shared<ACT_2>();
  auto output = boost::make_shared<Output>();  

  double lr = 0.01 / epoch;

  // Set optimizer (gradient descent)
  {
    using Optimizer = ffnn::optimizer::GradientDescent<FC_H1>;
    h1->setOptimizer(boost::make_shared<Optimizer>(lr));
  }
  {
    using Optimizer = ffnn::optimizer::GradientDescent<FC_H2>;
    h2->setOptimizer(boost::make_shared<Optimizer>(lr));
  }

  // Create network
  std::vector<Layer::Ptr> layers({input, h1, a1, h2, a2, output});

  std::stringstream issname;
  issname << "/home/brian/network_saves/990_5000.net";
  FFNN_INFO("Loading : " << issname.str());
  std::ifstream ifs(issname.str().c_str(), std::ios::binary);
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

  // Check that error montonically decreases
  float prev_error = std::numeric_limits<float>::infinity();
  for (size_t idx = 0UL; idx < iterations; idx++)
  {
    FFNN_INFO(idx << " " << epoch << " lr: " << lr);
    double error = 0;
    for (size_t jdx = 0UL; jdx < epoch; jdx++)
    {
      size_t kdx = jdx;

      // Forward activate
      (*input) << samples[kdx];
      for(const auto& layer : layers)
      {
        layer->forward();
      }
      Eigen::VectorXf netout(samples[kdx].rows(), 1);
      (*output) >> netout;

      // Compute error and check
      error += (targets[kdx] - netout).norm();

      // Set target
      (*output) << targets[kdx];

      // Backward propogated error
      for(const auto& layer : layers)
      {
        layer->backward();
      }
    }

    // Trigger optimization
    for(const auto& layer : layers)
    {
      layer->update();
    }

    // Save network
    if (!(idx % 10))
    {
      std::stringstream ssname;
      ssname << "/home/brian/network_saves/" << idx << "_" << epoch << ".net";
      FFNN_INFO("Saving : " << ssname.str());
      std::ofstream of(ssname.str().c_str(), std::ios::binary);
      for(const auto& layer : layers)
      {
        try
        {
          ffnn::save(of, *layer);
        }
        catch (const boost::archive::archive_exception& ex)
        {
          FFNN_ERROR(ex.what());
        }
      }
      of.close();
    }

    // Check for early-stopping
    FFNN_INFO("Error : " << error);
    if (std::abs(error - prev_error) < 1e-12)
    {
      FFNN_INFO("Training complete");
      return 0;
    }
    prev_error = error;
  }
  return 0;
}
