/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/layer/input.h>
#include <ffnn/layer/convolution.h>
#include <ffnn/layer/output.h>
#include <ffnn/layer/layer.h>


TEST(TestLayerConvolution, StaticInstanceColEmbedding_Forward)
{
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Convolution = ffnn::layer::Convolution<float, 16, 16, 3, 4, 4, 4, 1, ffnn::layer::ColEmbedding>;
  using Output = ffnn::layer::Output<float>;

  // Shape inferred from template args
  auto input = boost::make_shared<Input>(16 * 16 * 3);
  auto convolution = boost::make_shared<Convolution>();
  auto output = boost::make_shared<Output>();

  ffnn::layer::connect<Layer>(input, convolution);
  ffnn::layer::connect<Layer>(convolution, output);

  input->initialize();
  FFNN_INFO(input->inputShape());
  FFNN_INFO(input->outputShape());

  convolution->initialize();
  FFNN_INFO(convolution->inputShape());
  FFNN_INFO(convolution->outputShape());

  output->initialize();
  FFNN_INFO(output->inputShape());
  FFNN_INFO(output->outputShape());

  const auto& ish = convolution->inputShape();
  Eigen::VectorXf in_data(ish.size(), 1);
  in_data.setOnes();

  (*input) << in_data;
  input->forward();
  convolution->forward();
  output->forward();

  const auto& osh = convolution->outputShape();
  Eigen::VectorXf out_data(osh.size(), 1);
  out_data.setZero();
  (*output) >> out_data;

  //Eigen::Map<Eigen::MatrixXf> om(out_data.data(), osh.height, osh.width);
  //FFNN_INFO("\n" << om);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}