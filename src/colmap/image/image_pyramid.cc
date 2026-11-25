// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/image_pyramid.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace colmap {

namespace {

GrayscaleImageLevel Downsample2xBox(const GrayscaleImageLevel& prev) {
  GrayscaleImageLevel next;
  next.width = prev.width / 2;
  next.height = prev.height / 2;
  // Exact 2x2 box filter averages pixels [2*ix, 2*ix+1] whose full-res centers
  // sit at (2*ix + 1) / prev.scale_x, which is exactly (ix + 0.5) /
  // (prev.scale_x * 0.5), so halving scale preserves exact sub-pixel alignment
  // even when prev.width or prev.height is odd.
  next.scale_x = prev.scale_x * 0.5;
  next.scale_y = prev.scale_y * 0.5;
  next.data.resize(static_cast<size_t>(next.width) * next.height);

  for (int y = 0; y < next.height; ++y) {
    const uint8_t* row0 =
        prev.data.data() + static_cast<size_t>(2 * y) * prev.width;
    const uint8_t* row1 = row0 + prev.width;
    uint8_t* dst = next.data.data() + static_cast<size_t>(y) * next.width;
    for (int x = 0; x < next.width; ++x) {
      const int sx = 2 * x;
      const int sum =
          static_cast<int>(row0[sx]) + static_cast<int>(row0[sx + 1]) +
          static_cast<int>(row1[sx]) + static_cast<int>(row1[sx + 1]);
      dst[x] = static_cast<uint8_t>((sum + 2) >> 2);
    }
  }
  return next;
}

}  // namespace

GrayscaleImagePyramid GrayscaleImagePyramid::Build(const Bitmap& bitmap,
                                                   const int num_levels,
                                                   const int max_image_size) {
  THROW_CHECK(!bitmap.IsEmpty());
  THROW_CHECK_GE(num_levels, 1);

  const int orig_width = bitmap.Width();
  const int orig_height = bitmap.Height();

  const bool needs_resize =
      max_image_size > 0 && std::max(orig_width, orig_height) > max_image_size;
  if (bitmap.IsGrey() && !needs_resize) {
    return BuildFromGrayscale(orig_width,
                              orig_height,
                              orig_width,
                              orig_height,
                              bitmap.RowMajorData(),
                              num_levels);
  }

  Bitmap grey = bitmap.IsGrey() ? bitmap.Clone() : bitmap.CloneAsGrey();
  if (needs_resize) {
    grey.Thumbnail(max_image_size, Bitmap::RescaleFilter::kBox);
  }

  return BuildFromGrayscale(orig_width,
                            orig_height,
                            grey.Width(),
                            grey.Height(),
                            grey.RowMajorData(),
                            num_levels);
}

GrayscaleImagePyramid GrayscaleImagePyramid::BuildFromGrayscale(
    const int orig_width,
    const int orig_height,
    const int base_width,
    const int base_height,
    std::vector<uint8_t> base_data,
    const int num_levels) {
  THROW_CHECK_GT(orig_width, 0);
  THROW_CHECK_GT(orig_height, 0);
  THROW_CHECK_GT(base_width, 0);
  THROW_CHECK_GT(base_height, 0);
  THROW_CHECK_EQ(base_data.size(),
                 static_cast<size_t>(base_width) * base_height);
  THROW_CHECK_GE(num_levels, 1);

  GrayscaleImagePyramid pyramid;
  pyramid.orig_width_ = orig_width;
  pyramid.orig_height_ = orig_height;
  pyramid.levels_.reserve(num_levels);

  GrayscaleImageLevel base;
  base.width = base_width;
  base.height = base_height;
  base.scale_x =
      static_cast<double>(base_width) / static_cast<double>(orig_width);
  base.scale_y =
      static_cast<double>(base_height) / static_cast<double>(orig_height);
  base.data = std::move(base_data);
  pyramid.levels_.push_back(std::move(base));

  for (int lvl = 1; lvl < num_levels; ++lvl) {
    const GrayscaleImageLevel& prev = pyramid.levels_.back();
    if (prev.width < 4 || prev.height < 4) {
      break;
    }
    pyramid.levels_.push_back(Downsample2xBox(prev));
  }

  return pyramid;
}

const GrayscaleImageLevel& GrayscaleImagePyramid::Level(const int level) const {
  THROW_CHECK_GE(level, 0);
  THROW_CHECK_LT(level, static_cast<int>(levels_.size()));
  return levels_[level];
}

int GrayscaleImagePyramid::SelectLevelForSpacing(
    const double full_res_spacing) const {
  THROW_CHECK(!levels_.empty());
  if (!(full_res_spacing > 0.0) || !std::isfinite(full_res_spacing)) {
    return 0;
  }
  const double base_scale = 0.5 * (levels_[0].scale_x + levels_[0].scale_y);
  const double spacing_in_base = full_res_spacing * base_scale;
  if (spacing_in_base <= 1.0) {
    return 0;
  }
  const int target = static_cast<int>(std::llround(std::log2(spacing_in_base)));
  return std::clamp(target, 0, static_cast<int>(levels_.size()) - 1);
}

}  // namespace colmap
