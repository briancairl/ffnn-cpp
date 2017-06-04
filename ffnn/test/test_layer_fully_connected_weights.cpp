/**
 * @author Brian Cairl
 * @date 2017
 */
// C++ Standard Library
#include <fstream>
#include <type_traits>

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/fully_connected/weights.h>

TEST(TestLayerFullyConnectedWeights, Static_Sizing)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weights;

  EXPECT_NO_THROW(weights.setZero());

  EXPECT_EQ(weights.weights.rows(), 6);
  EXPECT_EQ(weights.weights.cols(), 4);
  EXPECT_EQ(weights.biases.rows(), 6);
  EXPECT_EQ(weights.biases.cols(), 1);
}

TEST(TestLayerFullyConnectedWeights, Static_Assignment)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weightsA;
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsB *= 2;
  weightsA *= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 2.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 2.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Static_ElementWiseMultiplication)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weightsA;
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA *= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 6.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 6.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Static_ElementWiseAddition)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weightsA;
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA += weightsB;

  EXPECT_NEAR(weightsA.biases(0), 5.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 5.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Static_ElementWiseSubtraction)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weightsA;
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA -= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 1.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 1.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Static_Serialization)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float, options<4, 6>> Weights;

  Weights weights;
  weights.setZero();
  {
    std::ofstream ofs("weights.bin", std::ios::binary);
    FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE oar(ofs);
    oar << weights;
  }

  Weights weights_loaded;
  {
    std::ifstream ifs("weights.bin", std::ios::binary);
    FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE iar(ifs);
    iar >> weights_loaded;
  }

  EXPECT_EQ(weights.weights.rows(), weights_loaded.weights.rows());
  EXPECT_EQ(weights.weights.cols(), weights_loaded.weights.cols());
}

TEST(TestLayerFullyConnectedWeights, Dynamic_Sizing)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weights;

  EXPECT_NO_THROW(weights.setZero(4, 6));

  EXPECT_EQ(weights.weights.rows(), 6);
  EXPECT_EQ(weights.weights.cols(), 4);
  EXPECT_EQ(weights.biases.rows(), 6);
  EXPECT_EQ(weights.biases.cols(), 1);
}

TEST(TestLayerFullyConnectedWeights, Dynamic_Assignment)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weightsA;
  weightsA.setZero(4, 6);
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.setZero(4, 6);
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsB *= 2;
  weightsA *= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 2.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 2.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Dynamic_ElementWiseMultiplication)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weightsA;
  weightsA.setZero(4, 6);
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.setZero(4, 6);
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA *= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 6.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 6.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Dynamic_ElementWiseAddition)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weightsA;
  weightsA.setZero(4, 6);
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.setZero(4, 6);
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA += weightsB;

  EXPECT_NEAR(weightsA.biases(0), 5.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 5.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Dynamic_ElementWiseSubtraction)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weightsA;
  weightsA.setZero(4, 6);
  weightsA.biases.setOnes();
  weightsA.weights.setOnes();

  Weights weightsB;
  weightsB.setZero(4, 6);
  weightsB.biases.setOnes();
  weightsB.weights.setOnes();

  weightsA *= 3;
  weightsB *= 2;
  weightsA -= weightsB;

  EXPECT_NEAR(weightsA.biases(0), 1.0, 1e-9);
  EXPECT_NEAR(weightsA.weights(0, 0), 1.0, 1e-9);
}

TEST(TestLayerFullyConnectedWeights, Dynamic_Serialization)
{
  using ffnn::layer::fully_connected::weights::options;
  typedef ffnn::layer::fully_connected::Weights<float> Weights;

  Weights weights;
  weights.setZero(4, 6);
  {
    std::ofstream ofs("weights.bin", std::ios::binary);
    FFNN_SERIALIZATION_OUTPUT_ARCHIVE_TYPE oar(ofs);
    oar << weights;
  }

  Weights weights_loaded;
  {
    std::ifstream ifs("weights.bin", std::ios::binary);
    FFNN_SERIALIZATION_INPUT_ARCHIVE_TYPE iar(ifs);
    iar >> weights_loaded;
  }

  EXPECT_EQ(weights.weights.rows(), weights_loaded.weights.rows());
  EXPECT_EQ(weights.weights.cols(), weights_loaded.weights.cols());
}


// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
