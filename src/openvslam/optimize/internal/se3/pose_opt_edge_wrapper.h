#ifndef OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
#define OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H

#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/radial_division.h"
#include "openvslam/camera/cube_space.h"
#include "openvslam/optimize/internal/se3/perspective_pose_opt_edge.h"
#include "openvslam/optimize/internal/se3/equirectangular_pose_opt_edge.h"
#include "openvslam/optimize/internal/se3/g2o_cubemap_vertices_edges.h"
#include "edge_extra.h"

#include <g2o/core/robust_kernel_impl.h>

namespace openvslam {

namespace data {
class landmark;
} // namespace data

namespace optimize {
namespace internal {
namespace se3 {

// [ivan] template removed
class pose_opt_edge_wrapper {
public:
    pose_opt_edge_wrapper() = delete;

    /**
     *
     * @param shot_vtx
     * @param pos_w
     * @param idx
     * @param obs_x
     * @param obs_y
     * @param obs_x_right
     * @param inv_sigma_sq
     * @param sqrt_chi_sq
     * @param extra Extra info, we take no ownership in this function
     */
    inline pose_opt_edge_wrapper(camera::base* camera, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                          const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                          const float inv_sigma_sq, const float sqrt_chi_sq,
                          void *extra = nullptr);

    virtual ~pose_opt_edge_wrapper() = default;


    void set_as_inlier() const {edge_->setLevel(0);}

    void set_as_outlier() const {edge_->setLevel(1);}

    // [ivan] this is never called, why?
    bool depth_is_positive() const;

    g2o::OptimizableGraph::Edge* edge_;

    camera::base* camera_;
    const unsigned int idx_;
    const bool is_monocular_;
};

inline pose_opt_edge_wrapper::pose_opt_edge_wrapper(camera::base* camera, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                                                       const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                       const float inv_sigma_sq, const float sqrt_chi_sq,
                                                       void *extra)
    : camera_(camera), idx_(idx), is_monocular_(obs_x_right < 0) {
    // 拘束条件を設定
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            auto c = static_cast<camera::perspective*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Fisheye: {
            auto c = static_cast<camera::fisheye*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Equirectangular: {
            assert(is_monocular_);

            auto c = static_cast<camera::equirectangular*>(camera_);

            auto edge = new equirectangular_pose_opt_edge();

            const Vec2_t obs{obs_x, obs_y};
            edge->setMeasurement(obs);
            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

            edge->cols_ = c->cols_;
            edge->rows_ = c->rows_;

            edge->pos_w_ = pos_w;

            edge->setVertex(0, shot_vtx);

            edge_ = edge;

            break;
        }
        case camera::model_type_t::RadialDivision: {
            auto c = static_cast<camera::radial_division*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::VirtualCube:{
            CubeSpaceExtra* cb_extra = static_cast<CubeSpaceExtra*>(extra);

            auto c = static_cast<camera::CubeSpace*>(camera_);
            // todo ivan. here we ignore stereo observation
            auto edge = new g2o::EdgeSE3ProjectXYZMultiPinholeOnlyPose();

            const Vec2_t obs{obs_x, obs_y};
            edge->setMeasurementInFace(obs);
            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

            edge->setCam(c);
            edge->setFace(cb_extra->face);
            edge->Xw = pos_w;
            edge->setPoseBC(cb_extra->T_B_C);

            edge->setVertex(0, shot_vtx);

            edge_ = edge;
            break;
        }
    }

    // loss functionを設定
    auto huber_kernel = new g2o::RobustKernelHuber();
    huber_kernel->setDelta(sqrt_chi_sq);
    edge_->setRobustKernel(huber_kernel);
}

inline bool pose_opt_edge_wrapper::depth_is_positive() const {
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Fisheye: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Equirectangular: {
            return true;
        }
        case camera::model_type_t::RadialDivision: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::VirtualCube: {
            AssertLog(false, "to implemented");
        }
    }

    return true;
}

} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
