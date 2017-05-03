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
#include <ffnn/layer/output.h>
#include <ffnn/layer/hidden.h>
#include <ffnn/layer/layer.h>

  using M1 = Eigen::Matrix<float, 2, 5,  Eigen::ColMajor>;
  using M2 = Eigen::Matrix<float, 5, 2, Eigen::RowMajor>;


class Hidden :
  public ffnn::layer::Hidden<float, -1, -1, M1, M2>
{
public:
  Hidden(int32_t i, int32_t j) :
    ffnn::layer::Hidden<float, -1, -1, M1, M2>(i, j)
  {}

  bool forward()
  {
    FFNN_ERROR("Input  : " << input_->rows()  << " by " << input_->cols());
    FFNN_ERROR("\n" << (*input_));
    FFNN_ERROR("Output : " << output_->rows() << " by " << output_->cols());
    FFNN_ERROR("\n" << (*output_));
    return true;
  }
};


/***********************************************************/
//
/***********************************************************/
TEST(TestLayerHiddenBasic, Mapping2D)
{

  // Layer-type alias
  using Layer  = ffnn::layer::Layer<float>;
  using Input  = ffnn::layer::Input<float>;
  using Output = ffnn::layer::Output<float>;


  // Layer sizes
  static const Layer::SizeType DIM = 10;

  // Create layers
  auto input = boost::make_shared<Input>(DIM);  
  auto hidden = boost::make_shared<Hidden>(DIM, DIM);
  auto output = boost::make_shared<Output>();  

  // Create network
  std::vector<Layer::Ptr> layers({input, hidden, output});

  // Connect layers
  for (size_t idx = 1UL; idx < layers.size(); idx++)
  {
    EXPECT_TRUE(ffnn::layer::connect<Layer>(layers[idx-1UL], layers[idx]));
  }

  // Initialize and check all layers and 
  for(const auto& layer : layers)
  {
    EXPECT_TRUE(layer->initialize());
    EXPECT_TRUE(layer->isInitialized());
  }

  M1 input_mat;
  for (size_t idx = 0; idx < DIM; idx++)
  {
    input_mat(idx) = idx;
  }
  (*input) << input_mat;

  input->forward();
  hidden->forward();

  FFNN_ERROR(output->inputSize());
  (*output) >> input_mat;
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
