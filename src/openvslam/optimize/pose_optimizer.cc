#include "openvslam/data/multi_frame.h"
#include "openvslam/data/landmark.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/optimize/internal/se3/pose_opt_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <vector>
#include <mutex>

#include <spdlog/spdlog.h>
#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

using namespace std;
namespace openvslam {
namespace optimize {

pose_optimizer::pose_optimizer(const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer::optimize(data::MultiFrame& frm) const {
    // 1. Construct an optimizer

    auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    unsigned int num_init_obs = 0;

    // 2. Convert the frame to the g2o vertex, then set it to the optimizer

    auto frm_vtx = new internal::se3::shot_vertex();
    frm_vtx->setId(frm.id_);
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(frm.cam_pose_cw_));
    frm_vtx->setFixed(false);
    optimizer.addVertex(frm_vtx);

    // unsigned int num_keypts = 0;
    // for (const auto& f : frm.frames) num_keypts += f.num_keypts_;


    // 3. Connect the landmark vertices by using projection edges

    // Container of the reprojection edges
    vector<vector<internal::se3::pose_opt_edge_wrapper>> pose_opt_edge_wraps;
    pose_opt_edge_wraps.resize(frm.frames.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    vector<vector<bool>> outlier_flags;
    outlier_flags.resize(frm.frames.size());

    for (int i=0; i<frm.frames.size(); ++i){
        auto &f = frm.frames[i];
        outlier_flags[i] = vector<bool>(f->num_keypts_, false);
        pose_opt_edge_wraps[i].reserve(f->num_keypts_);
        for (unsigned int idx = 0; idx < f->num_keypts_; ++idx) {
            const auto& lm = f->landmarks_.at(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            ++num_init_obs;

            // Connect the frame and the landmark vertices using the projection edges
            const auto& undist_keypt = f->undist_keypts_.at(idx);
            // todo ivan. stereo in multi-cam is not supported at this moment, but it should be easy.
            //  cases stereo would be proper: large overlap region, rgbd.
            const float x_right = f->stereo_x_right_.at(idx);
            const float inv_sigma_sq = f->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (f->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;

            AssertLog(f->camera_->model_type_ == camera::model_type_t::VirtualCube, "for now, only virtual cube is supported");
            const auto& cube_point = f->cube_keypts_.at(idx);
            internal::se3::CubeSpaceExtra extra{cube_point.face, frm.rig->poses_inv[i]};
            auto pose_opt_edge_wrap = internal::se3::pose_opt_edge_wrapper(
                    f->camera_, frm_vtx, lm->get_pos_in_world(),
                    idx, cube_point.u, cube_point.v, x_right,
                    inv_sigma_sq, sqrt_chi_sq, &extra);
            pose_opt_edge_wraps[i].push_back(pose_opt_edge_wrap);
            optimizer.addEdge(pose_opt_edge_wrap.edge_);
        }
    }

    if (num_init_obs < 5) {
        return 0;
    }

    // 4. Perform robust Bundle Adjustment (BA)

    unsigned int num_bad_obs = 0;
    for (unsigned int trial = 0; trial < num_trials_; ++trial) {
        optimizer.initializeOptimization();
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (int i = 0; i < frm.frames.size(); ++i) {
            for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps[i]) {
                auto edge = pose_opt_edge_wrap.edge_;

                if (outlier_flags[i].at(pose_opt_edge_wrap.idx_)) {
                    // outliers are not optimized, so call computeError manually to get err
                    edge->computeError();
                }

                // todo [ivan] why not check positive depth here??
                if (pose_opt_edge_wrap.is_monocular_) {
                    if (chi_sq_2D < edge->chi2()) {
                        outlier_flags[i].at(pose_opt_edge_wrap.idx_) = true;
                        pose_opt_edge_wrap.set_as_outlier();
                        ++num_bad_obs;
                    }
                    else {
                        outlier_flags[i].at(pose_opt_edge_wrap.idx_) = false;
                        pose_opt_edge_wrap.set_as_inlier();
                    }
                }
                else {
                    if (chi_sq_3D < edge->chi2()) {
                        outlier_flags[i].at(pose_opt_edge_wrap.idx_) = true;
                        pose_opt_edge_wrap.set_as_outlier();
                        ++num_bad_obs;
                    }
                    else {
                        outlier_flags[i].at(pose_opt_edge_wrap.idx_) = false;
                        pose_opt_edge_wrap.set_as_inlier();
                    }
                }

                if (trial == num_trials_ - 2) {
                    edge->setRobustKernel(nullptr);
                }
            }
        }

        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    // 5. Update the information

    // discard outliers
    for (int i = 0; i < frm.frames.size(); ++i) {
        for (int idx=0; idx<outlier_flags[i].size(); ++idx) {
            if (outlier_flags[i][idx]){
                frm.frames[i]->landmarks_.at(idx) = nullptr;
            }
            // [ivan] not call lm->increase_num_observed as it could be optimized more than once
        }
    }

    frm.set_cam_pose(frm_vtx->estimate());

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace openvslam
