// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/image/image_pyramid.h"

#include <cmath>
#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET 1
#define COLMAP_PHOTOMETRIC_AVX2_TARGET __attribute__((target("avx2,fma")))
#define COLMAP_PHOTOMETRIC_AVX2_INLINE \
  __attribute__((target("avx2,fma"), always_inline)) inline
#else
#define COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET 0
#endif

namespace colmap {
namespace photometric_simd {

inline bool CpuSupportsAvx2Fma() {
#if COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET
  static const bool supported =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return supported;
#else
  return false;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Scalar kernels (portable fallback & parity reference)
////////////////////////////////////////////////////////////////////////////////

inline bool InterpolateBilinearBatchScalar(const GrayscaleImageLevel& level,
                                           const double* xs,
                                           const double* ys,
                                           const int n,
                                           double* vals,
                                           double* dxs,
                                           double* dys) {
  const int width = level.width;
  const int height = level.height;
  const double max_u = static_cast<double>(width - 1);
  const double max_v = static_cast<double>(height - 1);
  const double scale_x = level.scale_x;
  const double scale_y = level.scale_y;
  const uint8_t* data = level.data.data();
  const bool compute_grad = (dxs != nullptr && dys != nullptr);

  for (int m = 0; m < n; ++m) {
    const double u = xs[m] * scale_x - 0.5;
    const double v = ys[m] * scale_y - 0.5;
    if (!(u >= 0.0 && u < max_u && v >= 0.0 && v < max_v)) {
      return false;
    }
    const int x0 = static_cast<int>(u);
    const int y0 = static_cast<int>(v);
    const double fx = u - static_cast<double>(x0);
    const double fy = v - static_cast<double>(y0);

    const uint8_t* row0 = data + static_cast<size_t>(y0) * width + x0;
    const uint8_t* row1 = row0 + width;
    const double i00 = static_cast<double>(row0[0]);
    const double i01 = static_cast<double>(row0[1]);
    const double i10 = static_cast<double>(row1[0]);
    const double i11 = static_cast<double>(row1[1]);

    const double v0 = i00 + fx * (i01 - i00);
    const double v1 = i10 + fx * (i11 - i10);
    vals[m] = v0 + fy * (v1 - v0);
    if (compute_grad) {
      dxs[m] = ((1.0 - fy) * (i01 - i00) + fy * (i11 - i10)) * scale_x;
      dys[m] = (v1 - v0) * scale_y;
    }
  }
  return true;
}

inline bool ComputeLSNCCResidualScalar(const double* vals,
                                       const double* src_norm,
                                       const int n,
                                       const double min_norm,
                                       double* p_norm,
                                       double* residuals,
                                       double* inv_sigma) {
  double sum = 0.0;
  for (int m = 0; m < n; ++m) {
    sum += vals[m];
  }
  const double mean = sum / static_cast<double>(n);

  double sq_norm = 0.0;
  for (int m = 0; m < n; ++m) {
    const double c = vals[m] - mean;
    p_norm[m] = c;
    sq_norm += c * c;
  }

  if (!(sq_norm >= min_norm * min_norm)) {
    return false;
  }

  const double inv_s = 1.0 / std::sqrt(sq_norm);
  *inv_sigma = inv_s;
  for (int m = 0; m < n; ++m) {
    const double pm = p_norm[m] * inv_s;
    p_norm[m] = pm;
    residuals[m] = pm - src_norm[m];
  }
  return true;
}

// Project a row-major [n x num_cols] raw Jacobian matrix G through the LS-NCC
// normalization Jacobian J_Psi = inv_sigma * (I - 1/n 11^T - p_norm p_norm^T).
// Supports g_row_major == out_jac when accumulate == false.
inline void ProjectLSNCCJacobianMatrixScalar(const double* g_row_major,
                                             const double* p_norm,
                                             const double inv_sigma,
                                             const int n,
                                             const int num_cols,
                                             double* out_jac,
                                             const bool accumulate) {
  const double inv_n = 1.0 / static_cast<double>(n);
  for (int c = 0; c < num_cols; ++c) {
    double sum = 0.0;
    double dot = 0.0;
    for (int m = 0; m < n; ++m) {
      const double g = g_row_major[static_cast<size_t>(m) * num_cols + c];
      sum += g;
      dot += p_norm[m] * g;
    }
    const double mean = sum * inv_n;
    if (accumulate) {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        out_jac[off] += inv_sigma * (g_row_major[off] - mean - dot * p_norm[m]);
      }
    } else {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        out_jac[off] = inv_sigma * (g_row_major[off] - mean - dot * p_norm[m]);
      }
    }
  }
}

// Fused 2-row chain rule + LS-NCC projection for camera parameter blocks:
//   G[m, p] = gx[m] * J_raw[m, 0, p] + gy[m] * J_raw[m, 1, p]
// followed by J_Psi * G into out_jac (row-major [n x num_cols]).
inline void ChainRuleAndProjectLSNCCScalar(const double* gx,
                                           const double* gy,
                                           const double* J_raw,
                                           const double* p_norm,
                                           const double inv_sigma,
                                           const int n,
                                           const int num_cols,
                                           double* scratch_row_major,
                                           double* out_jac,
                                           const bool accumulate) {
  double* g_dst = accumulate ? scratch_row_major : out_jac;
  for (int m = 0; m < n; ++m) {
    const double ax = gx[m];
    const double ay = gy[m];
    const double* jx = J_raw + static_cast<size_t>(m) * 2 * num_cols;
    const double* jy = jx + num_cols;
    double* row = g_dst + static_cast<size_t>(m) * num_cols;
    for (int p = 0; p < num_cols; ++p) {
      row[p] = ax * jx[p] + ay * jy[p];
    }
  }
  ProjectLSNCCJacobianMatrixScalar(
      g_dst, p_norm, inv_sigma, n, num_cols, out_jac, accumulate);
}

////////////////////////////////////////////////////////////////////////////////
// AVX2 + FMA kernels (compiled with target("avx2,fma") while keeping the rest
// of the binary portable)
////////////////////////////////////////////////////////////////////////////////

#if COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET

COLMAP_PHOTOMETRIC_AVX2_INLINE double HorizontalSum256d(const __m256d v) {
  const __m128d lo = _mm256_castpd256_pd128(v);
  const __m128d hi = _mm256_extractf128_pd(v, 1);
  const __m128d sum2 = _mm_add_pd(lo, hi);
  const __m128d sum1 = _mm_hadd_pd(sum2, sum2);
  return _mm_cvtsd_f64(sum1);
}

COLMAP_PHOTOMETRIC_AVX2_TARGET inline bool InterpolateAndLSNCCAvx2(
    const GrayscaleImageLevel& level,
    const double* xs,
    const double* ys,
    const double* src_norm,
    const int n,
    const double min_norm,
    const bool compute_grad,
    double* vals,
    double* dxs,
    double* dys,
    double* p_norm,
    double* residuals,
    double* inv_sigma) {
  const int width = level.width;
  const int height = level.height;
  const double max_u = static_cast<double>(width - 1);
  const double max_v = static_cast<double>(height - 1);
  const double scale_x = level.scale_x;
  const double scale_y = level.scale_y;
  const uint8_t* data = level.data.data();

  const __m256d v_scale_x = _mm256_set1_pd(scale_x);
  const __m256d v_scale_y = _mm256_set1_pd(scale_y);
  const __m256d v_half = _mm256_set1_pd(0.5);
  const __m256d v_zero = _mm256_setzero_pd();
  const __m256d v_one = _mm256_set1_pd(1.0);
  const __m256d v_max_u = _mm256_set1_pd(max_u);
  const __m256d v_max_v = _mm256_set1_pd(max_v);

  __m256d v_sum = _mm256_setzero_pd();
  int m = 0;
  for (; m + 4 <= n; m += 4) {
    const __m256d vx = _mm256_loadu_pd(xs + m);
    const __m256d vy = _mm256_loadu_pd(ys + m);
    const __m256d vu = _mm256_fmsub_pd(vx, v_scale_x, v_half);
    const __m256d vv = _mm256_fmsub_pd(vy, v_scale_y, v_half);

    // Ordered comparisons automatically reject NaN / Inf.
    const __m256d ok_u0 = _mm256_cmp_pd(vu, v_zero, _CMP_GE_OQ);
    const __m256d ok_u1 = _mm256_cmp_pd(vu, v_max_u, _CMP_LT_OQ);
    const __m256d ok_v0 = _mm256_cmp_pd(vv, v_zero, _CMP_GE_OQ);
    const __m256d ok_v1 = _mm256_cmp_pd(vv, v_max_v, _CMP_LT_OQ);
    const __m256d ok =
        _mm256_and_pd(_mm256_and_pd(ok_u0, ok_u1), _mm256_and_pd(ok_v0, ok_v1));
    if (_mm256_movemask_pd(ok) != 0xF) {
      return false;
    }

    const __m128i ix0 = _mm256_cvttpd_epi32(vu);
    const __m128i iy0 = _mm256_cvttpd_epi32(vv);
    const __m256d vfx = _mm256_sub_pd(vu, _mm256_cvtepi32_pd(ix0));
    const __m256d vfy = _mm256_sub_pd(vv, _mm256_cvtepi32_pd(iy0));

    // Extract integer indices directly from XMM registers (no store-forwarding
    // stall).
    const uint8_t* r0_0 = data +
                          static_cast<size_t>(_mm_cvtsi128_si32(iy0)) * width +
                          _mm_cvtsi128_si32(ix0);
    const uint8_t* r0_1 =
        data + static_cast<size_t>(_mm_extract_epi32(iy0, 1)) * width +
        _mm_extract_epi32(ix0, 1);
    const uint8_t* r0_2 =
        data + static_cast<size_t>(_mm_extract_epi32(iy0, 2)) * width +
        _mm_extract_epi32(ix0, 2);
    const uint8_t* r0_3 =
        data + static_cast<size_t>(_mm_extract_epi32(iy0, 3)) * width +
        _mm_extract_epi32(ix0, 3);

    const __m256d vi00 =
        _mm256_cvtepi32_pd(_mm_set_epi32(r0_3[0], r0_2[0], r0_1[0], r0_0[0]));
    const __m256d vi01 =
        _mm256_cvtepi32_pd(_mm_set_epi32(r0_3[1], r0_2[1], r0_1[1], r0_0[1]));
    const __m256d vi10 = _mm256_cvtepi32_pd(
        _mm_set_epi32(r0_3[width], r0_2[width], r0_1[width], r0_0[width]));
    const __m256d vi11 = _mm256_cvtepi32_pd(_mm_set_epi32(
        r0_3[width + 1], r0_2[width + 1], r0_1[width + 1], r0_0[width + 1]));

    const __m256d vd0 = _mm256_sub_pd(vi01, vi00);
    const __m256d vd1 = _mm256_sub_pd(vi11, vi10);
    const __m256d vv0 = _mm256_fmadd_pd(vfx, vd0, vi00);
    const __m256d vv1 = _mm256_fmadd_pd(vfx, vd1, vi10);
    const __m256d vdv = _mm256_sub_pd(vv1, vv0);
    const __m256d vval = _mm256_fmadd_pd(vfy, vdv, vv0);
    _mm256_storeu_pd(vals + m, vval);
    v_sum = _mm256_add_pd(v_sum, vval);

    if (compute_grad) {
      const __m256d v1m_fy = _mm256_sub_pd(v_one, vfy);
      const __m256d vdu = _mm256_fmadd_pd(vfy, vd1, _mm256_mul_pd(v1m_fy, vd0));
      _mm256_storeu_pd(dxs + m, _mm256_mul_pd(vdu, v_scale_x));
      _mm256_storeu_pd(dys + m, _mm256_mul_pd(vdv, v_scale_y));
    }
  }

  double sum = HorizontalSum256d(v_sum);
  for (; m < n; ++m) {
    const double u = xs[m] * scale_x - 0.5;
    const double v = ys[m] * scale_y - 0.5;
    if (!(u >= 0.0 && u < max_u && v >= 0.0 && v < max_v)) {
      return false;
    }
    const int x0 = static_cast<int>(u);
    const int y0 = static_cast<int>(v);
    const double fx = u - static_cast<double>(x0);
    const double fy = v - static_cast<double>(y0);
    const uint8_t* row0 = data + static_cast<size_t>(y0) * width + x0;
    const uint8_t* row1 = row0 + width;
    const double i00 = static_cast<double>(row0[0]);
    const double i01 = static_cast<double>(row0[1]);
    const double i10 = static_cast<double>(row1[0]);
    const double i11 = static_cast<double>(row1[1]);
    const double v0 = i00 + fx * (i01 - i00);
    const double v1 = i10 + fx * (i11 - i10);
    const double val = v0 + fy * (v1 - v0);
    vals[m] = val;
    sum += val;
    if (compute_grad) {
      dxs[m] = ((1.0 - fy) * (i01 - i00) + fy * (i11 - i10)) * scale_x;
      dys[m] = (v1 - v0) * scale_y;
    }
  }

  const double mean = sum / static_cast<double>(n);
  const __m256d v_mean = _mm256_set1_pd(mean);

  __m256d v_sq = _mm256_setzero_pd();
  m = 0;
  for (; m + 4 <= n; m += 4) {
    const __m256d vc = _mm256_sub_pd(_mm256_loadu_pd(vals + m), v_mean);
    _mm256_storeu_pd(p_norm + m, vc);
    v_sq = _mm256_fmadd_pd(vc, vc, v_sq);
  }
  double sq_norm = HorizontalSum256d(v_sq);
  for (; m < n; ++m) {
    const double c = vals[m] - mean;
    p_norm[m] = c;
    sq_norm += c * c;
  }

  if (!(sq_norm >= min_norm * min_norm)) {
    return false;
  }

  const double inv_s = 1.0 / std::sqrt(sq_norm);
  *inv_sigma = inv_s;
  const __m256d v_inv_s = _mm256_set1_pd(inv_s);
  m = 0;
  for (; m + 4 <= n; m += 4) {
    const __m256d vp = _mm256_mul_pd(_mm256_loadu_pd(p_norm + m), v_inv_s);
    _mm256_storeu_pd(p_norm + m, vp);
    const __m256d vr = _mm256_sub_pd(vp, _mm256_loadu_pd(src_norm + m));
    _mm256_storeu_pd(residuals + m, vr);
  }
  for (; m < n; ++m) {
    const double pm = p_norm[m] * inv_s;
    p_norm[m] = pm;
    residuals[m] = pm - src_norm[m];
  }
  return true;
}

COLMAP_PHOTOMETRIC_AVX2_INLINE void ProjectLSNCCJacobianMatrixAvx2Impl(
    const double* g_row_major,
    const double* p_norm,
    const double inv_sigma,
    const int n,
    const int num_cols,
    double* out_jac,
    const bool accumulate) {
  const double inv_n = 1.0 / static_cast<double>(n);
  const __m256d v_inv_n = _mm256_set1_pd(inv_n);
  const __m256d v_inv_s = _mm256_set1_pd(inv_sigma);

  int c = 0;
  for (; c + 4 <= num_cols; c += 4) {
    __m256d v_sum = _mm256_setzero_pd();
    __m256d v_dot = _mm256_setzero_pd();
    for (int m = 0; m < n; ++m) {
      const __m256d vr =
          _mm256_loadu_pd(g_row_major + static_cast<size_t>(m) * num_cols + c);
      const __m256d vpm = _mm256_set1_pd(p_norm[m]);
      v_sum = _mm256_add_pd(v_sum, vr);
      v_dot = _mm256_fmadd_pd(vpm, vr, v_dot);
    }
    const __m256d v_mean = _mm256_mul_pd(v_sum, v_inv_n);
    if (accumulate) {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        const __m256d vr = _mm256_loadu_pd(g_row_major + off);
        const __m256d vpm = _mm256_set1_pd(p_norm[m]);
        const __m256d diff =
            _mm256_fnmadd_pd(v_dot, vpm, _mm256_sub_pd(vr, v_mean));
        const __m256d prev = _mm256_loadu_pd(out_jac + off);
        _mm256_storeu_pd(out_jac + off, _mm256_fmadd_pd(v_inv_s, diff, prev));
      }
    } else {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        const __m256d vr = _mm256_loadu_pd(g_row_major + off);
        const __m256d vpm = _mm256_set1_pd(p_norm[m]);
        const __m256d diff =
            _mm256_fnmadd_pd(v_dot, vpm, _mm256_sub_pd(vr, v_mean));
        _mm256_storeu_pd(out_jac + off, _mm256_mul_pd(v_inv_s, diff));
      }
    }
  }
  for (; c < num_cols; ++c) {
    double sum = 0.0;
    double dot = 0.0;
    for (int m = 0; m < n; ++m) {
      const double g = g_row_major[static_cast<size_t>(m) * num_cols + c];
      sum += g;
      dot += p_norm[m] * g;
    }
    const double mean = sum * inv_n;
    if (accumulate) {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        out_jac[off] += inv_sigma * (g_row_major[off] - mean - dot * p_norm[m]);
      }
    } else {
      for (int m = 0; m < n; ++m) {
        const size_t off = static_cast<size_t>(m) * num_cols + c;
        out_jac[off] = inv_sigma * (g_row_major[off] - mean - dot * p_norm[m]);
      }
    }
  }
}

COLMAP_PHOTOMETRIC_AVX2_TARGET inline void ProjectLSNCCJacobianMatrixAvx2(
    const double* g_row_major,
    const double* p_norm,
    const double inv_sigma,
    const int n,
    const int num_cols,
    double* out_jac,
    const bool accumulate) {
  ProjectLSNCCJacobianMatrixAvx2Impl(
      g_row_major, p_norm, inv_sigma, n, num_cols, out_jac, accumulate);
}

COLMAP_PHOTOMETRIC_AVX2_TARGET inline void ChainRuleAndProjectLSNCCAvx2(
    const double* gx,
    const double* gy,
    const double* J_raw,
    const double* p_norm,
    const double inv_sigma,
    const int n,
    const int num_cols,
    double* scratch_row_major,
    double* out_jac,
    const bool accumulate) {
  double* g_dst = accumulate ? scratch_row_major : out_jac;
  for (int m = 0; m < n; ++m) {
    const double ax = gx[m];
    const double ay = gy[m];
    const __m256d vax = _mm256_set1_pd(ax);
    const __m256d vay = _mm256_set1_pd(ay);
    const double* jx = J_raw + static_cast<size_t>(m) * 2 * num_cols;
    const double* jy = jx + num_cols;
    double* row = g_dst + static_cast<size_t>(m) * num_cols;
    int p = 0;
    for (; p + 4 <= num_cols; p += 4) {
      const __m256d vjx = _mm256_loadu_pd(jx + p);
      const __m256d vjy = _mm256_loadu_pd(jy + p);
      _mm256_storeu_pd(row + p,
                       _mm256_fmadd_pd(vax, vjx, _mm256_mul_pd(vay, vjy)));
    }
    for (; p < num_cols; ++p) {
      row[p] = ax * jx[p] + ay * jy[p];
    }
  }
  ProjectLSNCCJacobianMatrixAvx2Impl(
      g_dst, p_norm, inv_sigma, n, num_cols, out_jac, accumulate);
}

#endif  // COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET

////////////////////////////////////////////////////////////////////////////////
// Runtime-dispatched entry points
////////////////////////////////////////////////////////////////////////////////

inline bool InterpolateAndLSNCC(const GrayscaleImageLevel& level,
                                const double* xs,
                                const double* ys,
                                const double* src_norm,
                                const int n,
                                const double min_norm,
                                const bool compute_grad,
                                double* vals,
                                double* dxs,
                                double* dys,
                                double* p_norm,
                                double* residuals,
                                double* inv_sigma,
                                [[maybe_unused]] const bool use_simd) {
#if COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET
  if (use_simd && CpuSupportsAvx2Fma()) {
    return InterpolateAndLSNCCAvx2(level,
                                   xs,
                                   ys,
                                   src_norm,
                                   n,
                                   min_norm,
                                   compute_grad,
                                   vals,
                                   dxs,
                                   dys,
                                   p_norm,
                                   residuals,
                                   inv_sigma);
  }
#endif
  if (!InterpolateBilinearBatchScalar(level,
                                      xs,
                                      ys,
                                      n,
                                      vals,
                                      compute_grad ? dxs : nullptr,
                                      compute_grad ? dys : nullptr)) {
    return false;
  }
  return ComputeLSNCCResidualScalar(
      vals, src_norm, n, min_norm, p_norm, residuals, inv_sigma);
}

inline void ProjectLSNCCJacobianMatrix(const double* g_row_major,
                                       const double* p_norm,
                                       const double inv_sigma,
                                       const int n,
                                       const int num_cols,
                                       double* out_jac,
                                       const bool accumulate,
                                       [[maybe_unused]] const bool use_simd) {
#if COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET
  if (use_simd && CpuSupportsAvx2Fma()) {
    ProjectLSNCCJacobianMatrixAvx2(
        g_row_major, p_norm, inv_sigma, n, num_cols, out_jac, accumulate);
    return;
  }
#endif
  ProjectLSNCCJacobianMatrixScalar(
      g_row_major, p_norm, inv_sigma, n, num_cols, out_jac, accumulate);
}

inline void ChainRuleAndProjectLSNCC(const double* gx,
                                     const double* gy,
                                     const double* J_raw,
                                     const double* p_norm,
                                     const double inv_sigma,
                                     const int n,
                                     const int num_cols,
                                     double* scratch_row_major,
                                     double* out_jac,
                                     const bool accumulate,
                                     [[maybe_unused]] const bool use_simd) {
#if COLMAP_PHOTOMETRIC_HAS_AVX2_TARGET
  if (use_simd && CpuSupportsAvx2Fma()) {
    ChainRuleAndProjectLSNCCAvx2(gx,
                                 gy,
                                 J_raw,
                                 p_norm,
                                 inv_sigma,
                                 n,
                                 num_cols,
                                 scratch_row_major,
                                 out_jac,
                                 accumulate);
    return;
  }
#endif
  ChainRuleAndProjectLSNCCScalar(gx,
                                 gy,
                                 J_raw,
                                 p_norm,
                                 inv_sigma,
                                 n,
                                 num_cols,
                                 scratch_row_major,
                                 out_jac,
                                 accumulate);
}

}  // namespace photometric_simd
}  // namespace colmap
