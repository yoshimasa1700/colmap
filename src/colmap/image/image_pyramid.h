// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/bitmap.h"

#include <cstdint>
#include <vector>

namespace colmap {

// Single level of a grayscale 8-bit image pyramid.
// Coordinates use COLMAP's convention where pixel [ix, iy] is centered at
// (ix + 0.5, iy + 0.5) in level coordinates. A full-resolution COLMAP image
// coordinate (full_x, full_y) maps to continuous level pixel index
//   u = full_x * scale_x - 0.5,  v = full_y * scale_y - 0.5.
struct GrayscaleImageLevel {
  int width = 0;
  int height = 0;
  double scale_x = 1.0;
  double scale_y = 1.0;
  std::vector<uint8_t> data;

  // Bilinearly interpolate intensity and analytical derivatives w.r.t.
  // full-resolution COLMAP image coordinates (full_x, full_y).
  // Returns false if the coordinate lies outside the valid interpolation
  // domain [0, width - 1) x [0, height - 1).
  inline bool InterpolateBilinear(const double full_x,
                                  const double full_y,
                                  double* value,
                                  double* dx = nullptr,
                                  double* dy = nullptr) const {
    const double u = full_x * scale_x - 0.5;
    const double v = full_y * scale_y - 0.5;
    if (!(u >= 0.0 && u < static_cast<double>(width - 1) && v >= 0.0 &&
          v < static_cast<double>(height - 1))) {
      return false;
    }

    const int x0 = static_cast<int>(u);
    const int y0 = static_cast<int>(v);
    const double fx = u - static_cast<double>(x0);
    const double fy = v - static_cast<double>(y0);

    const uint8_t* row0 = data.data() + static_cast<size_t>(y0) * width + x0;
    const uint8_t* row1 = row0 + width;
    const double i00 = static_cast<double>(row0[0]);
    const double i01 = static_cast<double>(row0[1]);
    const double i10 = static_cast<double>(row1[0]);
    const double i11 = static_cast<double>(row1[1]);

    const double v0 = i00 + fx * (i01 - i00);
    const double v1 = i10 + fx * (i11 - i10);
    *value = v0 + fy * (v1 - v0);

    if (dx != nullptr) {
      const double du = (1.0 - fy) * (i01 - i00) + fy * (i11 - i10);
      *dx = du * scale_x;
    }
    if (dy != nullptr) {
      const double dv = v1 - v0;
      *dy = dv * scale_y;
    }
    return true;
  }
};

// Multi-scale power-of-two grayscale image pyramid built by exact 2x2 box
// downsampling (preserving sub-pixel alignment across levels even for odd
// image dimensions).
class GrayscaleImagePyramid {
 public:
  GrayscaleImagePyramid() = default;

  // Build a pyramid with up to `num_levels` levels from a Bitmap (converted to
  // grayscale if RGB). If `max_image_size > 0` and max(width, height) exceeds
  // it, the base level is first rescaled to fit while preserving the mapping
  // from original full-resolution camera coordinates.
  static GrayscaleImagePyramid Build(const Bitmap& bitmap,
                                     int num_levels,
                                     int max_image_size = -1);

  // Build a pyramid directly from an already-grayscale base buffer.
  static GrayscaleImagePyramid BuildFromGrayscale(
      int orig_width,
      int orig_height,
      int base_width,
      int base_height,
      std::vector<uint8_t> base_data,
      int num_levels);

  int NumLevels() const { return static_cast<int>(levels_.size()); }
  bool IsEmpty() const { return levels_.empty(); }
  int OrigWidth() const { return orig_width_; }
  int OrigHeight() const { return orig_height_; }

  const GrayscaleImageLevel& Level(int level) const;

  // Select the pyramid level whose pixel spacing best matches
  // `full_res_spacing` (in full-resolution pixels) to minimize aliasing.
  int SelectLevelForSpacing(double full_res_spacing) const;

 private:
  int orig_width_ = 0;
  int orig_height_ = 0;
  std::vector<GrayscaleImageLevel> levels_;
};

}  // namespace colmap
