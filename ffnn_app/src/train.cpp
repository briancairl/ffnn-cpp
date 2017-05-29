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
#include <opencv2/opencv.hpp>

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
  cv::Mat img = cv::imread("/home/brian/Desktop/tacocat.png");

  cv::Mat f_img;
  img.convertTo(f_img, CV_32FC3);

  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Conv = ffnn::layer::Convolution<float, -1, -1, -1, -1, -1, -1, 1, ffnn::layer::ColEmbedding>;
  using FullyConnected = ffnn::layer::FullyConnected<float>;
  using Activation = ffnn::layer::Activation<float, ffnn::neuron::RectifiedLinear<float>>;
  using Output = ffnn::layer::Output<float>;

  // Layer sizes
  static const Layer::SizeType DIM = 128 * 128 * 3;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);
  auto conv = boost::make_shared<Conv>(Conv::ShapeType(128, 128, 3), 10, 10, 3, 4);
  auto act = boost::make_shared<Activation>();
  auto fc = boost::make_shared<FullyConnected>(1);
  //auto act_out = boost::make_shared<Activation>();
  auto output = boost::make_shared<Output>();

  FFNN_ERROR(conv->inputShape());

  // Set optimizer (gradient descent)
  conv->setOptimizer(boost::make_shared<ffnn::optimizer::GradientDescent<Conv>>(5e-9));
  fc->setOptimizer(boost::make_shared<ffnn::optimizer::GradientDescent<FullyConnected>>(5e-9));

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
  conv->initialize(ND(0, 10.0/ DIM / DIM), ND(0, 10.0/ DIM / DIM));
  act->initialize();
  fc->initialize(ND(0, 10.0 / DIM / DIM), ND(0, 10.0 / DIM / DIM));
  output->initialize();

  // Create windows for display grids
  cv::namedWindow("Kernel-0", CV_WINDOW_NORMAL);
  cv::namedWindow("Kernel-1", CV_WINDOW_NORMAL);
  cv::namedWindow("Kernel-2", CV_WINDOW_NORMAL);
  cv::namedWindow("Output", CV_WINDOW_NORMAL);
  cv::namedWindow("Conv Output", CV_WINDOW_NORMAL);


  // Check that error montonically decreases
  double prev_error = std::numeric_limits<double>::infinity();
  double error = 0;
  for (size_t idx = 0UL; idx < 1e9; idx++)
  {

    const double angle =  (180.0 / 4.0) * (double)(idx % 4) / 4.0;
    cv::Point2f src_center(f_img.cols/2.0F, f_img.rows/2.0F);
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(f_img, dst, rot_mat, f_img.size());

    // Creat eigen mapping
    Eigen::Map<Eigen::MatrixXf> input_img((float*)dst.data, dst.channels() * dst.rows, dst.cols);
    Eigen::MatrixXf target_signal(1, 1);
    Eigen::MatrixXf output_signal(1, 1);

    target_signal(0) = static_cast<double>(idx % 4)/4.0;

    // Forward activate
    (*input) << input_img;
    for(const auto& layer : layers)
    {
      layer->forward();
    }
    (*output) >> output_signal;

    // Compute error and check
    error += (target_signal - output_signal).norm();

    // Set target_signal
    (*output) << target_signal;

    // Backward propogated error
    for(const auto& layer : layers)
    {
      layer->backward();
    }

    // Trigget optimization
    if (!(idx%10))
    {
      FFNN_INFO(error - prev_error);

      for(const auto& layer : layers)
      {
        layer->update();
      }

      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv->getParameters().filters[0].kernel;
        cv::Mat kernel_img_cv(10, 10, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-0", kernel_img_cv_norm);
      }
      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv->getParameters().filters[1].kernel;
        cv::Mat kernel_img_cv(10, 10, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-1", kernel_img_cv_norm);
      }
      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv->getParameters().filters[2].kernel;
        cv::Mat kernel_img_cv(10, 10, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-2", kernel_img_cv_norm);
      }

      // Store previous error
      prev_error = error;
      error = 0;
    }

    {
      cv::Mat kernel_img_cv(conv->outputShape().height / 3, conv->outputShape().width, CV_32FC3, const_cast<float*>(fc->getInputBuffer().data()));
      cv::Mat kernel_img_cv_norm;
      cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
      cv::imshow("Conv Output", kernel_img_cv_norm);
    }
    cv::waitKey(1);
  }


  cv::imshow("Display window", img);
  cv::waitKey(0);
  return 0;
}
