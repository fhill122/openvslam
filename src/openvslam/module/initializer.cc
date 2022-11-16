#include "openvslam/config.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/initialize/bearing_vector.h"
#include "openvslam/initialize/perspective.h"
#include "openvslam/match/area.h"
#include "openvslam/module/initializer.h"
#include "openvslam/optimize/global_bundle_adjuster.h"
#include "openvslam/camera/fake_stereo.h"
#include "openvslam/data/multi_frame.h"

#include "spdlog/spdlog.h"
#include "ivtb/periodic_runner.h"
#include <nanoflann/nanoflann.h>

using namespace std;

namespace openvslam {
namespace module {

initializer::initializer(bool is_mono,
                         data::map_database* map_db,
                         const YAML::Node& yaml_node)
    : is_mono_(is_mono), map_db_(map_db),
      num_ransac_iters_(yaml_node["num_ransac_iterations"].as<unsigned int>(100)),
      min_num_triangulated_(yaml_node["num_min_triangulated_pts"].as<unsigned int>(50)),
      parallax_deg_thr_(yaml_node["parallax_deg_threshold"].as<float>(1.0)),
      reproj_err_thr_(yaml_node["reprojection_error_threshold"].as<float>(4.0)),
      num_ba_iters_(yaml_node["num_ba_iterations"].as<unsigned int>(20)),
      scaling_factor_(yaml_node["scaling_factor"].as<float>(1.0)),
      use_fixed_seed_(yaml_node["use_fixed_seed"].as<bool>(false)) {
    spdlog::debug("CONSTRUCT: module::initializer");
}

initializer::~initializer() {
    spdlog::debug("DESTRUCT: module::initializer");
}

void initializer::reset() {
    initializer_.reset(nullptr);
    state_ = initializer_state_t::NotReady;
    init_frm_id_ = 0;
}

initializer_state_t initializer::get_state() const {
    return state_;
}

std::vector<cv::KeyPoint> initializer::get_initial_keypoints() const {
    return init_frm_.keypts_;
}

std::vector<int> initializer::get_initial_matches() const {
    return init_matches_;
}

bool initializer::initialize(data::MultiFrame& curr_frm) {
    state_ = initializer_state_t::Initializing;
    createMapByTriangulate(curr_frm);

    // check the state is succeeded or not
    if (state_ == initializer_state_t::Succeeded) {
        init_frm_id_ = curr_frm.id_;
        return true;
    }
    else {
        return false;
    }
}

bool initializer::createMapByTriangulate(data::MultiFrame &curr_frm) {
    assert(state_ == initializer_state_t::Initializing);

    // create keyframe
    curr_frm.set_cam_pose(Mat44_t::Identity());
    auto curr_keyfrm = data::MultiKeyframe::Create(curr_frm, map_db_);

    vector<shared_ptr<data::landmark>> new_lms;
    new_lms.reserve(curr_frm.getNumKeypts());
    for (int i=0; i<curr_frm.size(); ++i){
        for (int j = 0; j < curr_frm.rig->overlaps[i].size(); ++j) {
            const auto &overlap = curr_frm.rig->overlaps[i][j];
            AssertLog(i==overlap.ind1, "");
            const cv::Mat& mask = overlap.mask;
            const auto& frame_1 = curr_frm[overlap.ind1];
            const auto& frame_2 = curr_frm[overlap.ind2];
            const auto& kf_1 = curr_keyfrm->at(overlap.ind1);
            const auto& kf_2 = curr_keyfrm->at(overlap.ind2);

            // filter by overlap
            vector<cv::Point2f> keypts_1;
            eigen_alloc_vector<Vec3_t> bearings_1;
            vector<int> indices_1;
            keypts_1.reserve(frame_1->num_keypts_);
            bearings_1.reserve(frame_1->num_keypts_);
            indices_1.reserve(frame_1->num_keypts_);
            for (int k=0; k< frame_1->num_keypts_; ++k){
                if (mask.at<unsigned char>(
                        (int)round(frame_1->keypts_[k].pt.y), (int)round(frame_1->keypts_[k].pt.x)) >0 ){
                    keypts_1.push_back(frame_1->keypts_[k].pt);
                    bearings_1.push_back(frame_1->bearings_[k]);
                    indices_1.push_back(k);
                }
            }

            // triangulate, and count num
            vector<cv::Point2f> keypts_2;
            vector<cv::Point3f> pts3d;
            Eigen::Matrix4d T_2_1 = curr_frm.rig->poses[overlap.ind2] * curr_frm.rig->poses_inv[overlap.ind1];
            FakeStereo::FindStereo(frame_1->img_, frame_2->img_, keypts_1, bearings_1,
                                   T_2_1.block<3,3>(0,0), T_2_1.block<3,1>(0,3),
                                   curr_frm.rig->cameras[overlap.ind1].get(),
                                   curr_frm.rig->cameras[overlap.ind2].get(),
                                   pts3d, keypts_2);
            AssertLog(pts3d.size() == keypts_1.size(), "");
            AssertLog(pts3d.size() == keypts_2.size(), "");

            // relates to exist orb keypoints
            constexpr float kRadiusSqr = 6;
            vector<int> matched_ind;
            matched_ind.resize(keypts_1.size());
            for (int k =0; k <keypts_1.size(); ++k){
                const int ind_1 = indices_1[k];

                if (keypts_2[k].x==-1 && keypts_2[k].y==-1){
                    matched_ind[k] = -1;
                    continue ;
                }

                // search near lk result
                vector<pair<int,float>> near_kpts = frame_2->radiusSearch(keypts_2[k].x, keypts_2[k].y, kRadiusSqr);
                if (near_kpts.empty()){
                    matched_ind[k] = -1;
                    continue ;
                }

                // only check the best distance here
                unsigned int best_hamm_dist = (match::HAMMING_DIST_THR_HIGH + match::HAMMING_DIST_THR_LOW) / 2;
                int best_ind = -1;
                for (auto ind_dist : near_kpts){
                    unsigned int dist = match::compute_descriptor_distance_32(
                                frame_1->descriptors_.row(ind_1), frame_2->descriptors_.row(ind_dist.first) );
                    if (dist < best_hamm_dist){
                        best_ind = ind_dist.first;
                        best_hamm_dist = dist;
                    }
                }
                matched_ind[k] = best_ind;

                if (best_ind>0){
                    // build a landmark, note body collides with world frame now
                    const Vec4_t pos_f1{pts3d[k].x, pts3d[k].y, pts3d[k].z, 1};
                    const Vec4_t pos_w = curr_frm.rig->poses_inv[overlap.ind1] * pos_f1;
                    auto lm = shared_ptr<data::landmark>(new data::landmark({pos_w.x(), pos_w.y(), pos_w.z()},
                                                             kf_1, map_db_));
                    RUN_N_TIMES(20,spdlog::debug("added a lm ({},{},{})", pts3d[k].x, pts3d[k].y, pts3d[k].z));

                    // set the associations to the new keyframe
                    lm->add_observation(kf_1, ind_1);
                    lm->add_observation(kf_2, best_ind);
                    kf_1->add_landmark(lm, ind_1);
                    kf_2->add_landmark(lm, best_ind);

                    // update the descriptor
                    lm->compute_descriptor();
                    // update the geometry
                    lm->update_normal_and_depth();

                    // set the 2D-3D associations to the current frame
                    frame_1->landmarks_.at(ind_1) = lm;
                    frame_2->landmarks_.at(best_ind) = lm;

                    new_lms.push_back(lm);
                }
            }

        }
    }

    if (new_lms.size()<min_num_triangulated_){
        spdlog::warn("insufficient 3d points for initialization: {} vs {}", new_lms.size(), min_num_triangulated_);
        return false;
    }

    // compute BoW representation
    for (auto &kf : curr_keyfrm->frames)
        kf->compute_bow();

    // add to the map DB
    map_db_->add_keyframe(curr_keyfrm);

    // update the frame statistics
    curr_frm.ref_keyfrm_ = curr_keyfrm;


    // add the landmark to the map DB
    for (const auto& lm : new_lms){
        map_db_->add_landmark(lm);
    }

    map_db_->origin_keyfrm_ = curr_keyfrm;

    spdlog::info("new map created with {} points: frame {}", map_db_->get_num_landmarks(), curr_frm.id_);
    state_ = initializer_state_t::Succeeded;
    return true;
}

// bool initializer::try_initialize_for_stereo(data::frame& curr_frm) {
//     assert(state_ == initializer_state_t::Initializing);
//     // count the number of valid depths
//     unsigned int num_valid_depths = std::count_if(curr_frm.depths_.begin(), curr_frm.depths_.end(),
//                                                   [](const float depth) {
//                                                       return 0 < depth;
//                                                   });
//     return min_num_triangulated_ <= num_valid_depths;
// }
//
// bool initializer::create_map_for_stereo(data::frame& curr_frm) {
//     assert(state_ == initializer_state_t::Initializing);
//
//     // create an initial keyframe
//     curr_frm.set_cam_pose(Mat44_t::Identity());
//     auto curr_keyfrm = data::keyframe::make_keyframe(curr_frm, map_db_);
//
//     // compute BoW representation
//     curr_keyfrm->compute_bow();
//
//     // add to the map DB
//     map_db_->add_keyframe(curr_keyfrm);
//
//     // update the frame statistics
//     curr_frm.ref_keyfrm_ = curr_keyfrm;
//
//     for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx) {
//         // add a new landmark if tht corresponding depth is valid
//         const auto z = curr_frm.depths_.at(idx);
//         if (z <= 0) {
//             continue;
//         }
//
//         // build a landmark
//         const Vec3_t pos_w = curr_frm.triangulate_stereo(idx);
//         // [ivan] have to use eigen allocator
//         auto lm = std::make_shared<data::landmark>(pos_w, curr_keyfrm, map_db_);
//
//         // set the associations to the new keyframe
//         lm->add_observation(curr_keyfrm, idx);
//         curr_keyfrm->add_landmark(lm, idx);
//
//         // update the descriptor
//         lm->compute_descriptor();
//         // update the geometry
//         lm->update_normal_and_depth();
//
//         // set the 2D-3D associations to the current frame
//         curr_frm.landmarks_.at(idx) = lm;
//         curr_frm.outlier_flags_.at(idx) = false;
//
//         // add the landmark to the map DB
//         map_db_->add_landmark(lm);
//     }
//
//     // set the origin keyframe
//     map_db_->origin_keyfrm_ = curr_keyfrm;
//
//     spdlog::info("new map created with {} points: frame {}", map_db_->get_num_landmarks(), curr_frm.id_);
//     state_ = initializer_state_t::Succeeded;
//     return true;
// }

} // namespace module
} // namespace openvslam
