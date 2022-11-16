#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/local_bundle_adjuster.h"
#include "openvslam/optimize/internal/landmark_vertex_container.h"
#include "openvslam/optimize/internal/se3/shot_vertex_container.h"
#include "openvslam/optimize/internal/se3/reproj_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <unordered_map>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

using namespace std;
namespace openvslam {
namespace optimize {

local_bundle_adjuster::local_bundle_adjuster(const unsigned int num_first_iter,
                                             const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter) {}

void local_bundle_adjuster::optimize(const std::shared_ptr<openvslam::data::MultiKeyframe>& curr_keyfrm,
                                     bool* const force_stop_flag) const {
    constexpr int kWindow = 60;

    // 1. Aggregate the local and fixed keyframes, and local landmarks

    // Correct the local keyframes of the current keyframe
    std::unordered_map<unsigned int, std::shared_ptr<data::MultiKeyframe>> local_keyfrms;

    local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    const auto curr_covisibilities = curr_keyfrm->at(0)->map_db_->getKeyframes<vector>(
                                            curr_keyfrm->id_-kWindow, curr_keyfrm->id_);
    for (const auto& local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
    }

    // Correct landmarks seen in local keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> local_lms;

    for (const auto& local_keyfrm : local_keyfrms) {
        for (const auto &kf : local_keyfrm.second->frames){
            const auto landmarks = kf->get_landmarks();
            for (const auto& local_lm : landmarks) {
                if (!local_lm) {
                    continue;
                }
                if (local_lm->will_be_erased()) {
                    continue;
                }

                // Avoid duplication
                // note [ivan] how would that help? search is done already with count
                if (local_lms.count(local_lm->id_)) {
                    continue;
                }

                local_lms[local_lm->id_] = local_lm;
            }
        }
    }

    // Fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::MultiKeyframe>> fixed_keyfrms;

    for (const auto& local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (const auto& obs : observations) {
            const shared_ptr<data::keyframe> fixed_keyfrm_s = obs.first.lock();
            if (!fixed_keyfrm_s) {
                continue;
            }

            shared_ptr<data::MultiKeyframe> fixed_keyfrm = fixed_keyfrm_s->parent_->shared_from_this();

            // Do not add if it's in the local keyframes
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // Avoid duplication
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. Construct an optimizer

    auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    // Container of the shot vertices
    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, local_keyfrms.size() + fixed_keyfrms.size());
    // Save the converted keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::MultiKeyframe>> all_keyfrms;

    // Set the local keyframes to the optimizer
    for (const auto& id_local_keyfrm_pair : local_keyfrms) {
        const auto& local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);
    }

    // Set the fixed keyframes to the optimizer
    for (const auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        const auto& fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);
    }

    // 4. Connect the vertices of the keyframe and the landmark by using an edge of reprojection constraint

    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, local_lms.size());

    // Container of the reprojection edges
    using reproj_edge_wrapper = internal::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (const auto& id_local_lm_pair : local_lms) {
        const auto local_lm = id_local_lm_pair.second;

        // Convert the landmark to the g2o vertex, then set to the optimizer
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);

        const auto observations = local_lm->get_observations();
        for (const auto& obs : observations) {
            const shared_ptr<data::keyframe> keyfrm = obs.first.lock();
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            shared_ptr<data::MultiKeyframe> keyfrm_m = keyfrm->parent_->shared_from_this();

            internal::se3::shot_vertex* keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm_m);
            const cv::KeyPoint& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;

            AssertLog(keyfrm->camera_->model_type_ == camera::model_type_t::VirtualCube, "for now, only virtual cube is supported");
            const auto& cube_point = keyfrm->cube_keypts_.at(idx);
            internal::se3::CubeSpaceExtra extra{cube_point.face, keyfrm_m->rig->poses_inv[keyfrm->sibling_idx_]};
            auto reproj_edge_wrap = reproj_edge_wrapper(
                        keyfrm, keyfrm->camera_, keyfrm_vtx,
                        local_lm, lm_vtx,
                        idx, cube_point.u, cube_point.v, x_right,
                        inv_sigma_sq, sqrt_chi_sq, true,
                        &extra);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);

        }
    }

    // 5. Perform the first optimization

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. Discard outliers, then perform the second optimization

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            const auto& local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
    }

    // 7. Count the outliers
    vector<int> outliers;
    outliers.reserve(reproj_edge_wraps.size());
    for (int i=0; i<reproj_edge_wraps.size(); ++i){
        auto edge = reproj_edge_wraps[i].edge_;

        const auto local_lm = reproj_edge_wraps[i].lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wraps[i].is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wraps[i].depth_is_positive()) {
                outliers.push_back(i);
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wraps[i].depth_is_positive()) {
                outliers.push_back(i);
            }
        }
    }

    // 8. Update the information

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (int i:outliers) {
            reproj_edge_wraps[i].shot_->erase_landmark_with_index(reproj_edge_wraps[i].idx_);
            reproj_edge_wraps[i].lm_->erase_observation(reproj_edge_wraps[i].shot_);
        }

        for (const auto& id_local_keyfrm_pair : local_keyfrms) {
            const auto& local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->setCamPose(keyfrm_vtx->estimate());
        }

        for (const auto& id_local_lm_pair : local_lms) {
            const auto& local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }
    }
}

} // namespace optimize
} // namespace openvslam
