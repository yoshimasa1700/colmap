// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/image_pyramid.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GrayscaleImagePyramid, LinearRampInterpolationAndGradients) {
  // Construct a 64x48 grayscale image with exact linear ramp I(x, y) = 2*x +
  // 3*y + 10 evaluated at COLMAP pixel centers (ix + 0.5, iy + 0.5).
  const int kWidth = 64;
  const int kHeight = 48;
  std::vector<uint8_t> data(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      data[y * kWidth + x] = static_cast<uint8_t>(2 * x + 3 * y + 10);
    }
  }

  const auto pyramid = GrayscaleImagePyramid::BuildFromGrayscale(
      kWidth, kHeight, kWidth, kHeight, data, /*num_levels=*/4);
  ASSERT_EQ(pyramid.NumLevels(), 4);

  // Check exact values and derivatives at every pyramid level.
  // Because 2x2 box downsampling preserves linear functions exactly (up to
  // integer rounding of <= 0.5), all levels should reproduce the same ramp!
  for (int lvl = 0; lvl < pyramid.NumLevels(); ++lvl) {
    const auto& level = pyramid.Level(lvl);
    EXPECT_EQ(level.width, kWidth >> lvl);
    EXPECT_EQ(level.height, kHeight >> lvl);
    EXPECT_DOUBLE_EQ(level.scale_x, 1.0 / (1 << lvl));
    EXPECT_DOUBLE_EQ(level.scale_y, 1.0 / (1 << lvl));

    const double px = 21.35;
    const double py = 17.65;
    double val = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    ASSERT_TRUE(level.InterpolateBilinear(px, py, &val, &dx, &dy));
    const double expected_val = 2.0 * (px - 0.5) + 3.0 * (py - 0.5) + 10.0;
    EXPECT_NEAR(val, expected_val, 0.75);
    EXPECT_NEAR(dx, 2.0, 0.15);
    EXPECT_NEAR(dy, 3.0, 0.15);

    // Finite-difference check on the level itself (exact piecewise bilinear).
    const double eps = 1e-4;
    double vp = 0.0;
    double vm = 0.0;
    ASSERT_TRUE(level.InterpolateBilinear(px + eps, py, &vp));
    ASSERT_TRUE(level.InterpolateBilinear(px - eps, py, &vm));
    EXPECT_NEAR(dx, (vp - vm) / (2.0 * eps), 1e-6);
    ASSERT_TRUE(level.InterpolateBilinear(px, py + eps, &vp));
    ASSERT_TRUE(level.InterpolateBilinear(px, py - eps, &vm));
    EXPECT_NEAR(dy, (vp - vm) / (2.0 * eps), 1e-6);
  }
}

TEST(GrayscaleImagePyramid, OddDimensionsNoSubpixelShift) {
  const int kWidth = 35;
  const int kHeight = 27;
  std::vector<uint8_t> data(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      data[y * kWidth + x] = static_cast<uint8_t>(4 * x + 2 * y + 5);
    }
  }

  const auto pyramid = GrayscaleImagePyramid::BuildFromGrayscale(
      kWidth, kHeight, kWidth, kHeight, data, /*num_levels=*/3);
  ASSERT_EQ(pyramid.NumLevels(), 3);
  EXPECT_EQ(pyramid.Level(1).width, 17);
  EXPECT_EQ(pyramid.Level(1).height, 13);
  EXPECT_DOUBLE_EQ(pyramid.Level(1).scale_x, 0.5);
  EXPECT_DOUBLE_EQ(pyramid.Level(1).scale_y, 0.5);

  double v0 = 0.0;
  double v1 = 0.0;
  ASSERT_TRUE(pyramid.Level(0).InterpolateBilinear(19.0, 13.0, &v0));
  ASSERT_TRUE(pyramid.Level(1).InterpolateBilinear(19.0, 13.0, &v1));
  EXPECT_NEAR(v0, v1, 0.75);
}

TEST(GrayscaleImagePyramid, BoundsAndSelectLevelForSpacing) {
  Bitmap bitmap(32, 32, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(100, 100, 100));
  const auto pyramid = GrayscaleImagePyramid::Build(
      bitmap, /*num_levels=*/4, /*max_image_size=*/16);
  EXPECT_EQ(pyramid.OrigWidth(), 32);
  EXPECT_EQ(pyramid.OrigHeight(), 32);
  EXPECT_EQ(pyramid.Level(0).width, 16);
  EXPECT_EQ(pyramid.Level(0).height, 16);
  EXPECT_DOUBLE_EQ(pyramid.Level(0).scale_x, 0.5);

  double val = 0.0;
  EXPECT_FALSE(pyramid.Level(0).InterpolateBilinear(0.1, 10.0, &val));
  EXPECT_FALSE(pyramid.Level(0).InterpolateBilinear(31.9, 10.0, &val));
  EXPECT_TRUE(pyramid.Level(0).InterpolateBilinear(16.0, 16.0, &val));
  EXPECT_DOUBLE_EQ(val, 100.0);

  // Base scale is 0.5, so 2.0 full-res pixels == 1.0 base pixel -> level 0.
  EXPECT_EQ(pyramid.SelectLevelForSpacing(1.0), 0);
  EXPECT_EQ(pyramid.SelectLevelForSpacing(2.0), 0);
  EXPECT_EQ(pyramid.SelectLevelForSpacing(4.0), 1);
  EXPECT_EQ(pyramid.SelectLevelForSpacing(8.0), 2);
}

}  // namespace
}  // namespace colmap
