/**
 * @author Brian Cairl
 * @date 2017
 */

// GTest
#include <gtest/gtest.h>

// Eigen
#include <Eigen/Dense>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/distribution/distribution.h>
#include <ffnn/distribution/normal.h>

TEST(TestDistribution, SetRandom_Normal)
{
  typedef Eigen::Matrix<float, 10, 10> Matrix;
  typedef ffnn::distribution::Normal<Matrix::Scalar> NormalDistribution;

  Matrix matrix;
  ffnn::distribution::setRandom(matrix, NormalDistribution(0, 1));

  EXPECT_NEAR(matrix.mean(), 0, 0.05);

  float std = (matrix.array() - matrix.mean()).abs().sum() / matrix.size();
  EXPECT_NEAR(std, 1, 0.25);
}

TEST(TestDistribution, SetRandom_StandardNormal)
{
  typedef Eigen::Matrix<float, 10, 10> Matrix;
  typedef ffnn::distribution::Normal<Matrix::Scalar> NormalDistribution;

  Matrix matrix;
  ffnn::distribution::setRandom(matrix, NormalDistribution());

  EXPECT_NEAR(matrix.mean(), 0, 0.05);

  float std = (matrix.array() - matrix.mean()).abs().sum() / matrix.size();
  EXPECT_NEAR(std, 1, 0.25);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
