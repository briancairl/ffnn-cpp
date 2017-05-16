/**
 * @author Brian Cairl
 * @date 2017
 */

// C++ Standard Library
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/activation.h>
#include <ffnn/layer/layer.h>
#include <ffnn/layer/input.h>
#include <ffnn/layer/fully_connected.h>
#include <ffnn/layer/convolution.h>
#include <ffnn/layer/output.h>
#include <ffnn/neuron/rectified_linear.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/distribution/normal.h>

// Run tests
int main(int argc, char** argv)
{
  cv::Mat img = cv::imread("/home/briancairl/Pictures/tacocat_ds.jpg");

  cv::Mat f_img;
  img.convertTo(f_img, CV_32FC3);
  Eigen::Map<Eigen::MatrixXf> map_img((float*)f_img.data, f_img.channels() * f_img.rows, f_img.cols);
  Eigen::MatrixXf out_img = map_img;

  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Conv = ffnn::layer::Convolution<float, -1, -1, -1, -1, -1, -1, 1, ffnn::layer::RowEmbedding>;
  using FullyConnected = ffnn::layer::FullyConnected<float>;
  using Activation = ffnn::layer::Activation<float, ffnn::neuron::RectifiedLinear<float>>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIM = 128 * 128 * 3;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);
  auto conv = boost::make_shared<Conv>(Conv::ShapeType(128, 128, 3), 31, 31, 5, 10);
  auto act = boost::make_shared<Activation>();
  auto fc = boost::make_shared<FullyConnected>(49152);
  //auto act_out = boost::make_shared<Activation>();
  auto output = boost::make_shared<Output>();

  FFNN_ERROR(conv->inputShape());

  // Set optimizer (gradient descent)
  conv->setOptimizer(boost::make_shared<ffnn::optimizer::GradientDescent<Conv>>(5e-8));
  fc->setOptimizer(boost::make_shared<ffnn::optimizer::GradientDescent<FullyConnected>>(5e-8));

  // Create network
  std::vector<Layer::Ptr> layers({input, conv, act, fc, /*act_out,*/ output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]);
  }

  using ND = ffnn::distribution::Normal<float>;

  // Intializer layers
  input->initialize();
  conv->initialize(ND(0, 1.0 / DIM), ND(0, 1.0 / DIM));
  act->initialize();
  fc->initialize(ND(0, 1.0 / DIM), ND(0, 1.0 / DIM));
  //act_out->initialize();
  output->initialize();

  // Check that error montonically decreases
  float prev_error = std::numeric_limits<float>::infinity();
  for (size_t idx = 0UL; idx < 5000; idx++)
  {
    // Forward activate
    (*input) << map_img;
    for(const auto& layer : layers)
    {
      layer->forward();
    }
    (*output) >> out_img;

    // Compute error and check
    double error = (map_img - out_img).norm();
    FFNN_INFO(error - prev_error);

    // Set target
    (*output) << map_img;

    // Backward propogated error
    for(const auto& layer : layers)
    {
      layer->backward();
    }

    // Trigget optimization
    for(const auto& layer : layers)
    {
      layer->update();
    }

    cv::Mat out_img_cv(128, 128, CV_32FC3, out_img.data());
    cv::normalize(out_img_cv, out_img_cv, 0, 1, cv::NORM_MINMAX, CV_32FC3);
    cv::imshow("Display window", out_img_cv);
    cv::waitKey(1);

    // Store previous error
    prev_error = error;
  }


  cv::imshow("Display window", img);
  cv::waitKey(0);
  return 0;
}
