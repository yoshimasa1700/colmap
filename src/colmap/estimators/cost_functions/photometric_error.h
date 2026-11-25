// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/photometric_simd.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/image/image_pyramid.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

// Maximum number of camera parameters across all COLMAP camera models
// (RadTanThinPrismFisheye / FullOpenCV / ThinPrismFisheye have 12 params).
constexpr int kPhotometricMaxCamParams = 16;

// Fixed per-landmark source patch data shared across all target views of the
// landmark at a given pyramid level.
struct PhotometricPatchData {
  // Full-resolution COLMAP pixel coordinates of the M patch samples in the
  // source image.
  std::vector<double> src_xs;
  std::vector<double> src_ys;

  // Precomputed zero-mean unit-norm source intensities Psi(I_src) of length M.
  std::vector<double> src_norm;

  // Initial undistorted source rays (u, v, w) at the start of the current
  // pyramid level (w = 1 for perspective models, unit sphere ray for spherical
  // models). Used directly when source intrinsics are constant or as a
  // thread-safe warm-start for Newton undistortion when variable.
  std::vector<Eigen::Vector3d> src_rays_init;

  // Snapshot of source camera parameters at the start of the current pyramid
  // level so Evaluate() can detect whether undistortion needs refinement.
  std::vector<double> src_cam_params_init;

  // Full-resolution anchor pixel (center of patch) used to reconstruct the
  // updated 3D point along the source ray after plane refinement.
  Eigen::Vector2d anchor_xy = Eigen::Vector2d::Zero();

  int NumSamples() const { return static_cast<int>(src_norm.size()); }
};

namespace photometric_internal {

// Compute J_q = d(R(q)^T * p) / dq (3x4 row-major, Eigen xyzw storage).
// Since R(q)^T = R(q^*) where q^* = (-qx, -qy, -qz, qw), apply
// QuaternionRotatePointWithJac(q^*, p) and negate the first 3 columns.
inline Eigen::Vector3d QuaternionInverseRotatePointWithJac(const double* q,
                                                           const double* pt,
                                                           double* J_out) {
  const double q_conj[4] = {-q[0], -q[1], -q[2], q[3]};
  Eigen::Vector3d res = QuaternionRotatePointWithJac(q_conj, pt, J_out);
  if (J_out != nullptr) {
    for (int row = 0; row < 3; ++row) {
      J_out[row * 4 + 0] = -J_out[row * 4 + 0];
      J_out[row * 4 + 1] = -J_out[row * 4 + 1];
      J_out[row * 4 + 2] = -J_out[row * 4 + 2];
    }
  }
  return res;
}

// Project a batch of M 3D points `pts_in_cam` using
// `CameraModel::ImgFromCamWithJac`. `J_params_out` (optional, size M * 2 *
// num_params) and `J_uvw_out` (optional, size M * 6) are stored contiguously
// per sample m.
template <typename CameraModel>
bool ProjectBatchImpl(const double* params,
                      const Eigen::Vector3d* pts_in_cam,
                      const int n,
                      double* xs_out,
                      double* ys_out,
                      double* J_params_out,
                      double* J_uvw_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  for (int m = 0; m < n; ++m) {
    double* jp = J_params_out ? (J_params_out + m * 2 * kNumParams) : nullptr;
    double* ju = J_uvw_out ? (J_uvw_out + m * 6) : nullptr;
    if (!CameraModel::ImgFromCamWithJac(params,
                                        pts_in_cam[m].x(),
                                        pts_in_cam[m].y(),
                                        pts_in_cam[m].z(),
                                        &xs_out[m],
                                        &ys_out[m],
                                        jp,
                                        ju)) {
      return false;
    }
  }
  return true;
}

inline bool ProjectBatch(const CameraModelId model_id,
                         const double* params,
                         const Eigen::Vector3d* pts_in_cam,
                         const int n,
                         double* xs_out,
                         double* ys_out,
                         double* J_params_out,
                         double* J_uvw_out) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)    \
  case CameraModel::model_id:             \
    return ProjectBatchImpl<CameraModel>( \
        params, pts_in_cam, n, xs_out, ys_out, J_params_out, J_uvw_out);
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return false;
}

// Unproject a batch of fixed source pixels (src_xs[m], src_ys[m]) to rays
// (u_m, v_m, 1) for perspective models using warm-started Gauss-Newton with
// ImgFromCamWithJac, and optionally compute the implicit-function Jacobian
//   d(u_m, v_m) / d(params) = -J_uv^{-1} * J_params   (2 x num_params per m).
// When `skip_newton` is true (source intrinsics unchanged from init), evaluates
// ImgFromCamWithJac once at `init_rays[m]` without Newton iterations.
// For spherical models (whose params are constant metadata), returns the
// initial unit rays directly with zero param Jacobian.
template <typename CameraModel>
bool UnprojectBatchWithJacImpl(const double* params,
                               const double* src_xs,
                               const double* src_ys,
                               const Eigen::Vector3d* init_rays,
                               const int n,
                               const bool skip_newton,
                               const bool compute_jac,
                               Eigen::Vector3d* rays_out,
                               double* d_uv_d_params_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  if constexpr (!std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                   CameraModel>) {
    for (int m = 0; m < n; ++m) {
      rays_out[m] = init_rays[m];
      if (compute_jac && d_uv_d_params_out != nullptr) {
        std::fill_n(
            d_uv_d_params_out + m * 2 * kNumParams, 2 * kNumParams, 0.0);
      }
    }
    return true;
  } else {
    constexpr int kMaxNewtonIters = 15;
    double J_uvw[6];
    double J_params[2 * kPhotometricMaxCamParams];
    double* jp_out =
        (compute_jac && d_uv_d_params_out != nullptr) ? J_params : nullptr;

    for (int m = 0; m < n; ++m) {
      double u = init_rays[m].x();
      double v = init_rays[m].y();
      const double target_x = src_xs[m];
      const double target_y = src_ys[m];
      // Scale pixel tolerance to 4 ULPs of the pixel coordinates so it stays
      // above double-precision rounding even at >6000 px while remaining tight
      // enough for small distortion perturbations near the image center.
      const double px_scale =
          std::max(1.0, std::abs(target_x) + std::abs(target_y));
      const double px_tol =
          4.0 * std::numeric_limits<double>::epsilon() * px_scale;
      const double tol_sq = px_tol * px_tol;

      if (skip_newton) {
        if (jp_out != nullptr) {
          double px = 0.0;
          double py = 0.0;
          if (!CameraModel::ImgFromCamWithJac(
                  params, u, v, 1.0, &px, &py, jp_out, J_uvw)) {
            return false;
          }
        }
      } else {
        bool converged = false;
        for (int iter = 0; iter < kMaxNewtonIters; ++iter) {
          double px = 0.0;
          double py = 0.0;
          if (!CameraModel::ImgFromCamWithJac(
                  params, u, v, 1.0, &px, &py, jp_out, J_uvw)) {
            break;
          }
          const double ex = px - target_x;
          const double ey = py - target_y;
          if (ex * ex + ey * ey <= tol_sq) {
            converged = true;
            break;
          }
          // J_uv is the leading 2x2 block of 2x3 row-major J_uvw:
          //   [a b; c d] = [J_uvw[0] J_uvw[1]; J_uvw[3] J_uvw[4]]
          const double a = J_uvw[0];
          const double b = J_uvw[1];
          const double c = J_uvw[3];
          const double d = J_uvw[4];
          const double det = a * d - b * c;
          if (!(std::abs(det) > 1e-12)) {
            break;
          }
          const double inv_det = 1.0 / det;
          const double du = -(d * ex - b * ey) * inv_det;
          const double dv = -(-c * ex + a * ey) * inv_det;
          u += du;
          v += dv;
          if (du * du + dv * dv < 1e-28) {
            converged = true;
            break;
          }
        }

        if (!converged) {
          // Fallback to model's full CamFromImg once before declaring failure.
          if (!CameraModel::CamFromImg(params, target_x, target_y, &u, &v)) {
            return false;
          }
          double px = 0.0;
          double py = 0.0;
          if (!CameraModel::ImgFromCamWithJac(
                  params, u, v, 1.0, &px, &py, jp_out, J_uvw)) {
            return false;
          }
          const double ex = px - target_x;
          const double ey = py - target_y;
          if (!(ex * ex + ey * ey < 1e-6)) {
            return false;
          }
        }
      }

      rays_out[m] = Eigen::Vector3d(u, v, 1.0);

      if (jp_out != nullptr) {
        const double a = J_uvw[0];
        const double b = J_uvw[1];
        const double c = J_uvw[3];
        const double d = J_uvw[4];
        const double det = a * d - b * c;
        if (!(std::abs(det) > 1e-12)) {
          return false;
        }
        const double inv_det = -1.0 / det;  // includes minus sign of IFT
        // -J_uv^{-1} = inv_det * [d, -b; -c, a]
        double* out_m = d_uv_d_params_out + m * 2 * kNumParams;
        for (int p = 0; p < kNumParams; ++p) {
          const double jpx = J_params[p];
          const double jpy = J_params[kNumParams + p];
          out_m[p] = inv_det * (d * jpx - b * jpy);
          out_m[kNumParams + p] = inv_det * (-c * jpx + a * jpy);
        }
      }
    }
    return true;
  }
}

inline bool UnprojectBatchWithJac(const CameraModelId model_id,
                                  const double* params,
                                  const double* src_xs,
                                  const double* src_ys,
                                  const Eigen::Vector3d* init_rays,
                                  const int n,
                                  const bool skip_newton,
                                  const bool compute_jac,
                                  Eigen::Vector3d* rays_out,
                                  double* d_uv_d_params_out) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                         \
  case CameraModel::model_id:                                  \
    return UnprojectBatchWithJacImpl<CameraModel>(params,      \
                                                  src_xs,      \
                                                  src_ys,      \
                                                  init_rays,   \
                                                  n,           \
                                                  skip_newton, \
                                                  compute_jac, \
                                                  rays_out,    \
                                                  d_uv_d_params_out);
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return false;
}

inline bool HasSourceCamChanged(const double* src_cam,
                                const std::vector<double>& src_cam_init,
                                const int num_params) {
  for (int p = 0; p < num_params; ++p) {
    if (src_cam[p] != src_cam_init[p]) {
      return true;
    }
  }
  return false;
}

// Parameter-block layout for PhotometricNCCCostFunction. Deduplicates shared
// cameras, frames, and rig sensors so Ceres never receives the same double*
// twice in one residual block.
struct SourceTargetBlockLayout {
  CameraModelId src_model_id = CameraModelId::kInvalid;
  CameraModelId tgt_model_id = CameraModelId::kInvalid;
  int src_num_cam_params = 0;
  int tgt_num_cam_params = 0;

  bool same_frame = false;
  bool src_is_ref_sensor = true;
  bool tgt_is_ref_sensor = true;
  bool same_sensor = false;
  bool same_camera = false;

  int plane_idx = 0;
  int src_rfw_idx = -1;
  int tgt_rfw_idx = -1;
  int src_sfr_idx = -1;
  int tgt_sfr_idx = -1;
  int src_cam_idx = -1;
  int tgt_cam_idx = -1;

  std::vector<int32_t> block_sizes;

  static SourceTargetBlockLayout Create(const CameraModelId src_model_id,
                                        const CameraModelId tgt_model_id,
                                        const frame_t src_frame_id,
                                        const frame_t tgt_frame_id,
                                        const sensor_t src_sensor_id,
                                        const sensor_t tgt_sensor_id,
                                        const bool src_is_ref_sensor,
                                        const bool tgt_is_ref_sensor,
                                        const camera_t src_camera_id,
                                        const camera_t tgt_camera_id) {
    SourceTargetBlockLayout layout;
    layout.src_model_id = src_model_id;
    layout.tgt_model_id = tgt_model_id;
    layout.src_num_cam_params =
        static_cast<int>(CameraModelNumParams(src_model_id));
    layout.tgt_num_cam_params =
        static_cast<int>(CameraModelNumParams(tgt_model_id));
    THROW_CHECK_LE(layout.src_num_cam_params, kPhotometricMaxCamParams);
    THROW_CHECK_LE(layout.tgt_num_cam_params, kPhotometricMaxCamParams);

    layout.same_frame = (src_frame_id == tgt_frame_id);
    layout.src_is_ref_sensor = src_is_ref_sensor;
    layout.tgt_is_ref_sensor = tgt_is_ref_sensor;
    layout.same_sensor = (!src_is_ref_sensor && !tgt_is_ref_sensor &&
                          src_sensor_id == tgt_sensor_id);
    layout.same_camera = (src_camera_id == tgt_camera_id);

    // Block 0: plane n in R^3.
    layout.plane_idx = 0;
    layout.block_sizes.push_back(3);

    if (!layout.same_frame) {
      layout.src_rfw_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
      layout.tgt_rfw_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }

    if (!layout.src_is_ref_sensor) {
      layout.src_sfr_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }
    if (!layout.tgt_is_ref_sensor) {
      if (layout.same_sensor) {
        layout.tgt_sfr_idx = layout.src_sfr_idx;
      } else {
        layout.tgt_sfr_idx = static_cast<int>(layout.block_sizes.size());
        layout.block_sizes.push_back(7);
      }
    }

    layout.src_cam_idx = static_cast<int>(layout.block_sizes.size());
    layout.block_sizes.push_back(layout.src_num_cam_params);
    if (layout.same_camera) {
      layout.tgt_cam_idx = layout.src_cam_idx;
    } else {
      layout.tgt_cam_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(layout.tgt_num_cam_params);
    }

    return layout;
  }

  std::vector<double*> BuildParameterPointers(double* plane_ptr,
                                              double* src_rfw_ptr,
                                              double* tgt_rfw_ptr,
                                              double* src_sfr_ptr,
                                              double* tgt_sfr_ptr,
                                              double* src_cam_ptr,
                                              double* tgt_cam_ptr) const {
    std::vector<double*> ptrs;
    ptrs.reserve(block_sizes.size());
    ptrs.push_back(plane_ptr);
    if (!same_frame) {
      ptrs.push_back(src_rfw_ptr);
      ptrs.push_back(tgt_rfw_ptr);
    }
    if (!src_is_ref_sensor) {
      ptrs.push_back(src_sfr_ptr);
    }
    if (!tgt_is_ref_sensor && !same_sensor) {
      ptrs.push_back(tgt_sfr_ptr);
    }
    ptrs.push_back(src_cam_ptr);
    if (!same_camera) {
      ptrs.push_back(tgt_cam_ptr);
    }
    THROW_CHECK_EQ(ptrs.size(), block_sizes.size());
    return ptrs;
  }
};

// Unpacked frame & rig poses and composed relative transform
// T_ji = T_j * T_i^-1 for PhotometricNCCCostFunction.
struct RelativePoseState {
  const double* src_rfw = nullptr;
  const double* tgt_rfw = nullptr;
  const double* src_sfr = nullptr;
  const double* tgt_sfr = nullptr;

  Eigen::Matrix3d R_fi = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_fi = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_fj = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_fj = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_si = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_si = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_sj = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_sj = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_j = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_ji = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_ji = Eigen::Vector3d::Zero();

  static RelativePoseState FromParameters(const SourceTargetBlockLayout& layout,
                                          double const* const* parameters) {
    static constexpr double kIdentityPose[7] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    RelativePoseState s;
    s.src_rfw =
        layout.same_frame ? kIdentityPose : parameters[layout.src_rfw_idx];
    s.tgt_rfw =
        layout.same_frame ? kIdentityPose : parameters[layout.tgt_rfw_idx];
    s.src_sfr = layout.src_is_ref_sensor ? kIdentityPose
                                         : parameters[layout.src_sfr_idx];
    s.tgt_sfr = layout.tgt_is_ref_sensor ? kIdentityPose
                                         : parameters[layout.tgt_sfr_idx];

    if (!layout.same_frame) {
      s.R_fi = EigenQuaternionMap<double>(s.src_rfw).toRotationMatrix();
      s.t_fi = EigenVector3Map<double>(s.src_rfw + 4);
      s.R_fj = EigenQuaternionMap<double>(s.tgt_rfw).toRotationMatrix();
      s.t_fj = EigenVector3Map<double>(s.tgt_rfw + 4);
    }
    if (!layout.src_is_ref_sensor) {
      s.R_si = EigenQuaternionMap<double>(s.src_sfr).toRotationMatrix();
      s.t_si = EigenVector3Map<double>(s.src_sfr + 4);
    }
    if (!layout.tgt_is_ref_sensor) {
      s.R_sj = EigenQuaternionMap<double>(s.tgt_sfr).toRotationMatrix();
      s.t_sj = EigenVector3Map<double>(s.tgt_sfr + 4);
    }

    const Eigen::Matrix3d R_i = s.R_si * s.R_fi;
    const Eigen::Vector3d t_i = s.R_si * s.t_fi + s.t_si;
    s.R_j = s.R_sj * s.R_fj;
    const Eigen::Vector3d t_j = s.R_sj * s.t_fj + s.t_sj;
    s.R_ji = s.R_j * R_i.transpose();
    s.t_ji = t_j - s.R_ji * t_i;
    return s;
  }
};

struct BlockJacobianPtrs {
  double* plane = nullptr;
  double* src_rfw = nullptr;
  double* tgt_rfw = nullptr;
  double* src_sfr = nullptr;
  double* tgt_sfr = nullptr;
  double* src_cam = nullptr;
  double* tgt_cam = nullptr;

  static BlockJacobianPtrs FromJacobians(const SourceTargetBlockLayout& layout,
                                         double** jacobians) {
    BlockJacobianPtrs j;
    if (jacobians == nullptr) {
      return j;
    }
    j.plane = jacobians[layout.plane_idx];
    if (!layout.same_frame) {
      j.src_rfw = jacobians[layout.src_rfw_idx];
      j.tgt_rfw = jacobians[layout.tgt_rfw_idx];
    }
    if (!layout.src_is_ref_sensor) {
      j.src_sfr = jacobians[layout.src_sfr_idx];
    }
    if (!layout.tgt_is_ref_sensor) {
      j.tgt_sfr = jacobians[layout.tgt_sfr_idx];
    }
    j.src_cam = jacobians[layout.src_cam_idx];
    j.tgt_cam = jacobians[layout.tgt_cam_idx];
    return j;
  }

  bool NeedAny() const {
    return plane || src_rfw || tgt_rfw || src_sfr || tgt_sfr || src_cam ||
           tgt_cam;
  }
  bool NeedJUvw() const {
    return plane || src_rfw || tgt_rfw || src_sfr || tgt_sfr || src_cam;
  }
  bool NeedPoses() const { return src_rfw || tgt_rfw || src_sfr || tgt_sfr; }
};

// Small stack-backed buffer with heap fallback for large patch sizes.
template <typename T, size_t kStackCapacity>
class InlineBuffer {
 public:
  T* Resize(const size_t count) {
    if (count <= kStackCapacity) {
      return stack_;
    }
    heap_.resize(count);
    return heap_.data();
  }

 private:
  alignas(32) T stack_[kStackCapacity];
  std::vector<T> heap_;
};

}  // namespace photometric_internal

// Analytical Least-Squares NCC (Woodford 2018) photometric bundle adjustment
// cost function using LSPBA's (Woodford & Rosten 2020) source-anchored plane
// parameterization n in R^3.
//
// Residual vector (length M = patch->NumSamples()):
//   r = Psi(I_tgt(x')) - Psi(I_src(x))
// where ||r||^2 = 2 * (1 - NCC) in [0, 4].
class PhotometricNCCCostFunction : public ceres::CostFunction {
 public:
  PhotometricNCCCostFunction(
      const PhotometricPatchData* patch,
      const GrayscaleImageLevel* target_level,
      photometric_internal::SourceTargetBlockLayout layout,
      const double min_target_norm,
      const bool use_simd)
      : patch_(THROW_CHECK_NOTNULL(patch)),
        target_level_(THROW_CHECK_NOTNULL(target_level)),
        layout_(std::move(layout)),
        min_target_norm_(min_target_norm),
        use_simd_(use_simd) {
    const int num_samples = patch_->NumSamples();
    THROW_CHECK_GE(num_samples, 4);
    set_num_residuals(num_samples);
    *mutable_parameter_block_sizes() = layout_.block_sizes;
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const int M = patch_->NumSamples();
    const int num_blocks = static_cast<int>(layout_.block_sizes.size());

    auto SetBarrierAndZeroJacobians = [&]() {
      // Worst-case ZNSSD residual: ||r||^2 = 4.0 (NCC = -1), zero gradient.
      // Prevents Levenberg-Marquardt from being rewarded for pushing patches
      // out of bounds, behind a camera, or into textureless regions.
      const double barrier = 2.0 / std::sqrt(static_cast<double>(M));
      std::fill_n(residuals, M, barrier);
      if (jacobians != nullptr) {
        for (int b = 0; b < num_blocks; ++b) {
          if (jacobians[b] != nullptr) {
            std::fill_n(jacobians[b],
                        static_cast<size_t>(M) * layout_.block_sizes[b],
                        0.0);
          }
        }
      }
      return true;
    };

    const double* n_ptr = parameters[layout_.plane_idx];
    const Eigen::Vector3d n(n_ptr[0], n_ptr[1], n_ptr[2]);
    const double* src_cam = parameters[layout_.src_cam_idx];
    const double* tgt_cam = parameters[layout_.tgt_cam_idx];

    const auto pose = photometric_internal::RelativePoseState::FromParameters(
        layout_, parameters);
    const auto jac = photometric_internal::BlockJacobianPtrs::FromJacobians(
        layout_, jacobians);
    const bool need_any_jac = jac.NeedAny();
    const bool need_J_uvw = jac.NeedJUvw();

    // Step 1: Source unprojection (fast path skips Newton when src_cam is
    // unchanged from level init).
    const bool src_cam_changed = photometric_internal::HasSourceCamChanged(
        src_cam, patch_->src_cam_params_init, layout_.src_num_cam_params);

    constexpr size_t kStackMaxM = 64;
    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> rays_buf;
    photometric_internal::
        InlineBuffer<double, kStackMaxM * 2 * kPhotometricMaxCamParams>
            d_uv_buf;
    const Eigen::Vector3d* src_rays = patch_->src_rays_init.data();
    double* d_uv_d_src_cam = nullptr;

    if (src_cam_changed || jac.src_cam != nullptr) {
      Eigen::Vector3d* rays_ptr = rays_buf.Resize(M);
      if (jac.src_cam != nullptr) {
        d_uv_d_src_cam = d_uv_buf.Resize(static_cast<size_t>(M) * 2 *
                                         layout_.src_num_cam_params);
      }
      if (!photometric_internal::UnprojectBatchWithJac(
              layout_.src_model_id,
              src_cam,
              patch_->src_xs.data(),
              patch_->src_ys.data(),
              patch_->src_rays_init.data(),
              M,
              !src_cam_changed,
              jac.src_cam != nullptr,
              rays_ptr,
              d_uv_d_src_cam)) {
        return SetBarrierAndZeroJacobians();
      }
      src_rays = rays_ptr;
    }

    // Step 2: Warp all M patch rays through plane homography
    // H_ji = R_ji + t_ji * n^T.
    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> pts_tgt_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> alphas_buf;
    Eigen::Vector3d* pts_tgt = pts_tgt_buf.Resize(M);
    double* alphas = alphas_buf.Resize(M);

    const Eigen::Matrix3d H_ji = pose.R_ji + pose.t_ji * n.transpose();
    for (int m = 0; m < M; ++m) {
      const double alpha = n.dot(src_rays[m]);
      if (!(alpha > 1e-8)) {
        return SetBarrierAndZeroJacobians();
      }
      alphas[m] = alpha;
      pts_tgt[m] = H_ji * src_rays[m];
    }

    // Step 3: Batch target camera projection (+ analytic camera Jacobians).
    photometric_internal::InlineBuffer<double, kStackMaxM * 6> proj_buf;
    double* tgt_xs = proj_buf.Resize(static_cast<size_t>(M) * 6);
    double* tgt_ys = tgt_xs + M;
    double* tgt_vals = tgt_ys + M;
    double* tgt_dxs = tgt_vals + M;
    double* tgt_dys = tgt_dxs + M;
    double* p_norm = tgt_dys + M;

    photometric_internal::
        InlineBuffer<double, kStackMaxM * 2 * kPhotometricMaxCamParams>
            J_tgt_cam_buf;
    double* J_tgt_cam_raw =
        jac.tgt_cam != nullptr
            ? J_tgt_cam_buf.Resize(static_cast<size_t>(M) * 2 *
                                   layout_.tgt_num_cam_params)
            : nullptr;

    photometric_internal::InlineBuffer<double, kStackMaxM * 6> J_uvw_buf;
    double* J_uvw_raw =
        need_J_uvw ? J_uvw_buf.Resize(static_cast<size_t>(M) * 6) : nullptr;

    if (!photometric_internal::ProjectBatch(layout_.tgt_model_id,
                                            tgt_cam,
                                            pts_tgt,
                                            M,
                                            tgt_xs,
                                            tgt_ys,
                                            J_tgt_cam_raw,
                                            J_uvw_raw)) {
      return SetBarrierAndZeroJacobians();
    }

    // Step 4 & 5: Fused batch bilinear interpolation (+ image gradients) and
    // Least-Squares NCC normalization r = Psi(I_tgt) - s_k.
    double inv_sigma = 0.0;
    if (!photometric_simd::InterpolateAndLSNCC(*target_level_,
                                               tgt_xs,
                                               tgt_ys,
                                               patch_->src_norm.data(),
                                               M,
                                               min_target_norm_,
                                               need_any_jac,
                                               tgt_vals,
                                               tgt_dxs,
                                               tgt_dys,
                                               p_norm,
                                               residuals,
                                               &inv_sigma,
                                               use_simd_)) {
      return SetBarrierAndZeroJacobians();
    }

    if (!need_any_jac) {
      return true;
    }

    // Step 6: Chain rule g_y[m] = [dI/dx, dI/dy] * J_uvw[m] (1x3 per sample)
    // and row-major SIMD projection J_Psi * G directly into Ceres Jacobians.
    photometric_internal::InlineBuffer<Eigen::RowVector3d, kStackMaxM> gy_buf;
    Eigen::RowVector3d* gy = gy_buf.Resize(M);
    if (need_J_uvw) {
      for (int m = 0; m < M; ++m) {
        const double ix = tgt_dxs[m];
        const double iy = tgt_dys[m];
        const double* ju = J_uvw_raw + m * 6;
        gy[m] = Eigen::RowVector3d(ix * ju[0] + iy * ju[3],
                                   ix * ju[1] + iy * ju[4],
                                   ix * ju[2] + iy * ju[5]);
      }
    }

    photometric_internal::InlineBuffer<double,
                                       kStackMaxM * kPhotometricMaxCamParams>
        scratch_buf;

    // 6a. Plane Jacobian: d(y_m)/dn = t_ji * src_ray_m^T  (3 cols, in-place).
    if (jac.plane != nullptr) {
      for (int m = 0; m < M; ++m) {
        const double s = gy[m].dot(pose.t_ji);
        double* row = jac.plane + m * 3;
        row[0] = s * src_rays[m].x();
        row[1] = s * src_rays[m].y();
        row[2] = s * src_rays[m].z();
      }
      photometric_simd::ProjectLSNCCJacobianMatrix(
          jac.plane, p_norm, inv_sigma, M, 3, jac.plane, false, use_simd_);
    }

    // 6b-6e. Frame & rig pose Jacobians.
    if (jac.NeedPoses()) {
      ComputePoseJacobians(
          pose, jac, src_rays, alphas, gy, p_norm, inv_sigma, M, scratch_buf);
    }

    // 6f. Target camera intrinsics Jacobian.
    if (jac.tgt_cam != nullptr) {
      photometric_simd::ChainRuleAndProjectLSNCC(tgt_dxs,
                                                 tgt_dys,
                                                 J_tgt_cam_raw,
                                                 p_norm,
                                                 inv_sigma,
                                                 M,
                                                 layout_.tgt_num_cam_params,
                                                 nullptr,
                                                 jac.tgt_cam,
                                                 false,
                                                 use_simd_);
    }

    // 6g. Source camera intrinsics Jacobian (via implicit function theorem).
    // Accumulates into the shared block when same_camera is true.
    if (jac.src_cam != nullptr) {
      const int P_src = layout_.src_num_cam_params;
      photometric_internal::InlineBuffer<double, kStackMaxM * 2> guv_buf;
      double* gu = guv_buf.Resize(static_cast<size_t>(M) * 2);
      double* gv = gu + M;
      const Eigen::Matrix<double, 3, 2> H_uv = H_ji.leftCols<2>();
      for (int m = 0; m < M; ++m) {
        const Eigen::RowVector2d g_uv = gy[m] * H_uv;
        gu[m] = g_uv[0];
        gv[m] = g_uv[1];
      }
      const bool accumulate = layout_.same_camera && (jac.tgt_cam != nullptr);
      photometric_simd::ChainRuleAndProjectLSNCC(
          gu,
          gv,
          d_uv_d_src_cam,
          p_norm,
          inv_sigma,
          M,
          P_src,
          accumulate ? scratch_buf.Resize(static_cast<size_t>(M) * P_src)
                     : nullptr,
          jac.src_cam,
          accumulate,
          use_simd_);
    }

    return true;
  }

 private:
  void ComputePoseJacobians(
      const photometric_internal::RelativePoseState& pose,
      const photometric_internal::BlockJacobianPtrs& jac,
      const Eigen::Vector3d* src_rays,
      const double* alphas,
      const Eigen::RowVector3d* gy,
      const double* p_norm,
      const double inv_sigma,
      const int M,
      photometric_internal::InlineBuffer<double, 64 * kPhotometricMaxCamParams>&
          scratch_buf) const {
    constexpr size_t kStackMaxM = 64;
    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> p_fi_buf;
    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> p_w_buf;
    Eigen::Vector3d* p_fi = p_fi_buf.Resize(M);
    Eigen::Vector3d* p_w = p_w_buf.Resize(M);

    const Eigen::Matrix3d R_si_T = pose.R_si.transpose();
    const Eigen::Matrix3d R_fi_T = pose.R_fi.transpose();
    for (int m = 0; m < M; ++m) {
      p_fi[m] = layout_.src_is_ref_sensor
                    ? src_rays[m]
                    : (R_si_T * (src_rays[m] - alphas[m] * pose.t_si));
      p_w[m] = layout_.same_frame
                   ? p_fi[m]
                   : (R_fi_T * (p_fi[m] - alphas[m] * pose.t_fi));
    }

    double Jq[12];
    if (jac.tgt_rfw != nullptr) {
      for (int m = 0; m < M; ++m) {
        const Eigen::RowVector3d g_fj =
            layout_.tgt_is_ref_sensor ? gy[m] : (gy[m] * pose.R_sj);
        QuaternionRotatePointWithJac(pose.tgt_rfw, p_w[m].data(), Jq);
        double* row = jac.tgt_rfw + m * 7;
        for (int c = 0; c < 4; ++c) {
          row[c] = g_fj[0] * Jq[c] + g_fj[1] * Jq[4 + c] + g_fj[2] * Jq[8 + c];
        }
        row[4] = alphas[m] * g_fj[0];
        row[5] = alphas[m] * g_fj[1];
        row[6] = alphas[m] * g_fj[2];
      }
      photometric_simd::ProjectLSNCCJacobianMatrix(
          jac.tgt_rfw, p_norm, inv_sigma, M, 7, jac.tgt_rfw, false, use_simd_);
    }

    if (jac.src_rfw != nullptr) {
      for (int m = 0; m < M; ++m) {
        const Eigen::RowVector3d g_w = gy[m] * pose.R_j;
        const Eigen::RowVector3d g_fi_row = g_w * R_fi_T;
        const Eigen::Vector3d p_shift = p_fi[m] - alphas[m] * pose.t_fi;
        photometric_internal::QuaternionInverseRotatePointWithJac(
            pose.src_rfw, p_shift.data(), Jq);
        double* row = jac.src_rfw + m * 7;
        for (int c = 0; c < 4; ++c) {
          row[c] = g_w[0] * Jq[c] + g_w[1] * Jq[4 + c] + g_w[2] * Jq[8 + c];
        }
        row[4] = -alphas[m] * g_fi_row[0];
        row[5] = -alphas[m] * g_fi_row[1];
        row[6] = -alphas[m] * g_fi_row[2];
      }
      photometric_simd::ProjectLSNCCJacobianMatrix(
          jac.src_rfw, p_norm, inv_sigma, M, 7, jac.src_rfw, false, use_simd_);
    }

    if (jac.tgt_sfr != nullptr) {
      for (int m = 0; m < M; ++m) {
        const Eigen::Vector3d p_fj =
            layout_.same_frame ? p_fi[m]
                               : (pose.R_fj * p_w[m] + alphas[m] * pose.t_fj);
        QuaternionRotatePointWithJac(pose.tgt_sfr, p_fj.data(), Jq);
        double* row = jac.tgt_sfr + m * 7;
        for (int c = 0; c < 4; ++c) {
          row[c] =
              gy[m][0] * Jq[c] + gy[m][1] * Jq[4 + c] + gy[m][2] * Jq[8 + c];
        }
        row[4] = alphas[m] * gy[m][0];
        row[5] = alphas[m] * gy[m][1];
        row[6] = alphas[m] * gy[m][2];
      }
      photometric_simd::ProjectLSNCCJacobianMatrix(
          jac.tgt_sfr, p_norm, inv_sigma, M, 7, jac.tgt_sfr, false, use_simd_);
    }

    if (jac.src_sfr != nullptr) {
      const bool accumulate = layout_.same_sensor && (jac.tgt_sfr != nullptr);
      double* g_dst = accumulate
                          ? scratch_buf.Resize(static_cast<size_t>(M) * 7)
                          : jac.src_sfr;
      for (int m = 0; m < M; ++m) {
        const Eigen::RowVector3d g_fi_row =
            layout_.same_frame ? Eigen::RowVector3d(gy[m] * pose.R_sj)
                               : Eigen::RowVector3d(gy[m] * pose.R_j * R_fi_T);
        const Eigen::RowVector3d g_si_row = g_fi_row * R_si_T;
        const Eigen::Vector3d p_shift = src_rays[m] - alphas[m] * pose.t_si;
        photometric_internal::QuaternionInverseRotatePointWithJac(
            pose.src_sfr, p_shift.data(), Jq);
        double* row = g_dst + m * 7;
        for (int c = 0; c < 4; ++c) {
          row[c] = g_fi_row[0] * Jq[c] + g_fi_row[1] * Jq[4 + c] +
                   g_fi_row[2] * Jq[8 + c];
        }
        row[4] = -alphas[m] * g_si_row[0];
        row[5] = -alphas[m] * g_si_row[1];
        row[6] = -alphas[m] * g_si_row[2];
      }
      photometric_simd::ProjectLSNCCJacobianMatrix(
          g_dst, p_norm, inv_sigma, M, 7, jac.src_sfr, accumulate, use_simd_);
    }
  }

  const PhotometricPatchData* patch_;
  const GrayscaleImageLevel* target_level_;
  const photometric_internal::SourceTargetBlockLayout layout_;
  const double min_target_norm_;
  const bool use_simd_;
};

}  // namespace colmap
