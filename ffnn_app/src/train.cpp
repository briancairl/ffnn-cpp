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
#include <ffnn/neuron/leaky_rectified_linear.h>
#include <ffnn/optimizer/gradient_descent.h>
#include <ffnn/distribution/normal.h>

// Run tests
int main(int argc, char** argv)
{
  using namespace ffnn::distribution;
  using namespace ffnn::layer;
  using namespace ffnn::optimizer;
  cv::Mat img = cv::imread("/home/briancairl/Desktop/tacocat.jpg");

  cv::Mat f_img;
  img.convertTo(f_img, CV_32FC3);

  // Layer-type alias
  using Layer  = Layer<float>;
  using Input  = Input<float>;
  using Conv = Convolution<float>;
  using FullyConnected = FullyConnected<float>;
  using Activation = Activation<float, ffnn::neuron::LeakyRectifiedLinear<float>>;
  using Output = Output<float>;

  typedef GradientDescent<Conv, CrossEntropy> ConvOpt;
  typedef GradientDescent<FullyConnected, CrossEntropy> FCOpt;

  // Layer sizes
  static const ffnn::size_type DIM = 128 * 128 * 3;
  static const ffnn::size_type FDIM = 10;
  static const ffnn::size_type OUTRES = 20;


  // Create layers
  auto input = boost::make_shared<Input>(DIM);
  auto conv1 = boost::make_shared<Conv>(Conv::Configuration()
                                        .setInputShape(128, 128, 3)
                                        .setFilterShape(FDIM, FDIM, 3)
                                        .setStride(FDIM/2,FDIM/2)
                                        .setOptimizer(boost::make_shared<ConvOpt>(5e-7))
                                        .setParameterDistribution(boost::make_shared<Normal<float>>(0, 1e-4)));
  // auto conv2 = boost::make_shared<Conv>(Conv::Configuration()
  //                                       .setInputShape(40, 40, 3)
  //                                       .setFilterShape(5, 5, 3)
  //                                       .setStride(2,2)
  //                                       .setOptimizer(boost::make_shared<ConvOpt>(5e-7))
  //                                       .setParameterDistribution(boost::make_shared<Normal<float>>(0, 1e-4)));
  auto act1 = boost::make_shared<Activation>();
  //auto act2 = boost::make_shared<Activation>();
  auto fc = boost::make_shared<FullyConnected>(FullyConnected::Configuration()
                                               .setOutputShape(OUTRES)
                                               .setOptimizer(boost::make_shared<FCOpt>(5e-7))
                                               .setParameterDistribution(boost::make_shared<Normal<float>>(0, 1e-4)));
  auto output = boost::make_shared<Output>();

  FFNN_ERROR(conv1->getInputShape());

  // Create network
  //std::vector<Layer::Ptr> layers({input, conv1, act1, conv2, act2, fc, output});
  std::vector<Layer::Ptr> layers({input, conv1, act1, fc, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    connect<Layer>(layers[idx-1UL], layers[idx]);
  }

  //using ND = ffnn::distribution::Normal<float>;

  // Intializer layers
  for (auto& layer : layers)
  {
    layer->initialize();
  }

  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    FFNN_ERROR((layers[idx-1UL]->getOutputShape().size() == layers[idx]->getInputShape().size()));
    FFNN_INFO(layers[idx-1UL]->getOutputShape());
  }

  // Create windows for display grids
  cv::namedWindow("Kernel-0", CV_WINDOW_NORMAL);
  cv::namedWindow("Kernel-1", CV_WINDOW_NORMAL);
  cv::namedWindow("Kernel-2", CV_WINDOW_NORMAL);
  cv::namedWindow("Output", CV_WINDOW_NORMAL);
  cv::namedWindow("Conv-1", CV_WINDOW_NORMAL);
  cv::namedWindow("Conv-2", CV_WINDOW_NORMAL);

  // Check that error montonically decreases
  double prev_error = std::numeric_limits<double>::infinity();
  double error = 0;
  for (size_t idx = 0UL; idx < 1e9; idx++)
  {
    const double angle =  180.0 * (double)(idx % OUTRES) / static_cast<float>(OUTRES);
    cv::Point2f src_center(f_img.cols/2.0F, f_img.rows/2.0F);
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(f_img, dst, rot_mat, f_img.size());

    // Creat eigen mapping
    Eigen::Map<Eigen::MatrixXf> input_img((float*)dst.data, dst.channels() * dst.rows, dst.cols);
    Eigen::MatrixXf target_signal(OUTRES, 1);
    Eigen::MatrixXf output_signal(OUTRES, 1);

    target_signal.setZero();
    target_signal(idx % OUTRES) = 1.0;

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
    if (!(idx%OUTRES))
    {
      FFNN_INFO(error - prev_error);

      for(const auto& layer : layers)
      {
        layer->update();
      }

      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv1->getParameters()[0];
        cv::Mat kernel_img_cv(FDIM, FDIM, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-0", kernel_img_cv_norm);
      }
      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv1->getParameters()[1];
        cv::Mat kernel_img_cv(FDIM, FDIM, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-1", kernel_img_cv_norm);
      }
      {
        Eigen::Matrix<float, -1, -1, Eigen::ColMajor> ok = conv1->getParameters()[2];
        cv::Mat kernel_img_cv(FDIM, FDIM, CV_32FC3, const_cast<float*>(ok.data()));
        cv::Mat kernel_img_cv_norm;
        cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
        cv::imshow("Kernel-2", kernel_img_cv_norm);
      }

      // Store previous error
      prev_error = error;
      error = 0;
    }
    {
      cv::Mat kernel_img_cv(conv1->getOutputShape().height / 3, conv1->getOutputShape().width, CV_32FC3, const_cast<float*>(act1->getInputBuffer().data()));
      cv::Mat kernel_img_cv_norm;
      cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
      cv::imshow("Conv-1", kernel_img_cv_norm);
    }
    // {
    //   cv::Mat kernel_img_cv(conv2->getOutputShape().height / 3, conv2->getOutputShape().width, CV_32FC3, const_cast<float*>(fc->getInputBuffer().data()));
    //   cv::Mat kernel_img_cv_norm;
    //   cv::normalize(kernel_img_cv, kernel_img_cv_norm, 0, 1, cv::NORM_MINMAX, CV_32FC3);
    //   cv::imshow("Conv-2", kernel_img_cv_norm);
    // }
    cv::waitKey(1);
  }


  cv::imshow("Display window", img);
  cv::waitKey(0);
  return 0;
}
