/**
 * @author Brian Cairl
 * @date 2017
 */

// GTest
#include <gtest/gtest.h>

// FFNN
#include <ffnn/logging.h>
#include <ffnn/layer/shape.h>

TEST(TestLayerShape, ShapeAtCompileTime)
{
  typedef ffnn::layer::ShapeAtCompileTime<int, 1, 2, 3> TestShape;

  const auto h = TestShape::HeightAtCompileTime;
  const auto w = TestShape::WidthAtCompileTime;
  const auto d = TestShape::DepthAtCompileTime;
  EXPECT_EQ(h, 1);
  EXPECT_EQ(w, 2);
  EXPECT_EQ(d, 3);
}

TEST(TestLayerShape, Shape)
{
  const ffnn::layer::Shape<int> shape(1, 2, 3);
  EXPECT_EQ(shape.height, 1);
  EXPECT_EQ(shape.width,  2);
  EXPECT_EQ(shape.depth,  3);
}

// Run tests
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
