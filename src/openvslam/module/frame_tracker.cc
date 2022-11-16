#include "openvslam/camera/base.h"
#include "openvslam/data/multi_frame.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/bow_tree.h"
#include "openvslam/match/projection.h"
#include "openvslam/match/robust.h"
#include "openvslam/module/frame_tracker.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace module {

frame_tracker::frame_tracker(const unsigned int num_matches_thr)
    : num_matches_thr_(num_matches_thr), pose_optimizer_() {}

bool frame_tracker::motion_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                                       const Mat44_t& velocity) const {
    match::projection projection_matcher(0.9, true);

    // Set the initial pose by using the motion model
    curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);

    // Initialize the 2D-3D matches
    for (auto &frame : curr_frm.frames)
        std::fill(frame->landmarks_.begin(), frame->landmarks_.end(), nullptr);

    // Reproject the 3D points observed in the last frame and find 2D-3D matches
    const float margin = 20;
    unsigned int num_matches = 0;
    for (int i=0; i< curr_frm.frames.size(); ++i){
        num_matches += projection_matcher.match_current_and_last_frames(
                                                *curr_frm.frames[i], *last_frm.frames[i], margin);
    }

    if (num_matches < num_matches_thr_) {
        // Increment the margin, and search again
        num_matches = 0;
        for (int i=0; i< curr_frm.frames.size(); ++i){
            std::fill(curr_frm.frames[i]->landmarks_.begin(), curr_frm.frames[i]->landmarks_.end(), nullptr);
            num_matches += projection_matcher.match_current_and_last_frames(
                *curr_frm.frames[i], *last_frm.frames[i], margin);
        }
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    auto num_valid_matches = pose_optimizer_.optimize(curr_frm);
    
    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::bow_match_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                                          const std::shared_ptr<data::MultiKeyframe>& ref_keyfrm) const {
    match::bow_tree bow_matcher(0.7, true);

    // Compute the BoW representations to perform the BoW match
    for (auto &f : curr_frm.frames) f->compute_bow();

    // Search 2D-2D matches between the ref keyframes and the current frame
    // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
    unsigned int num_matches = 0;
    for (int i=0; i<curr_frm.frames.size(); ++i){
        std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
        num_matches += bow_matcher.match_frame_and_keyframe(
                    ref_keyfrm->frames[i], *curr_frm.frames[i], matched_lms_in_curr);
        // Update the 2D-3D matches
        curr_frm.frames[i]->landmarks_ = matched_lms_in_curr;

    }
    if (num_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    // The initial value is the pose of the previous frame
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    auto num_valid_matches = pose_optimizer_.optimize(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

// todo ivan. at this stage, I think ransac instead of optimization should be used
bool frame_tracker::robust_match_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                                             const std::shared_ptr<data::MultiKeyframe>& ref_keyfrm) const {
    match::robust robust_matcher(0.8, false);

    unsigned int num_matches = 0;

    for (int i=0; i<curr_frm.frames.size(); ++i){
        // Search 2D-2D matches between the ref keyframes and the current frame
        // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
        std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
        num_matches += robust_matcher.match_frame_and_keyframe(
                            *curr_frm.frames[i], ref_keyfrm->frames[i], matched_lms_in_curr);

        // Update the 2D-3D matches
        curr_frm.frames[i]->landmarks_ = matched_lms_in_curr;
    }
    if (num_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    // The initial value is the pose of the previous frame
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    auto num_valid_matches = pose_optimizer_.optimize(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

// unsigned int frame_tracker::discard_outliers(data::MultiFrame& curr_frm) const {
//     unsigned int num_valid_matches = 0;
//
//     for (auto &f : curr_frm.frames) {
//         for (unsigned int idx = 0; idx < f.num_keypts_; ++idx) {
//             if (!f.landmarks_.at(idx)) {
//                 continue;
//             }
//
//             auto& lm = f.landmarks_.at(idx);
//
//             if (f.outlier_flags_.at(idx)) {
//                 f.outlier_flags_.at(idx) = false;
//                 lm->is_observable_in_tracking_ = false;
//                 lm->identifier_in_local_lm_search_ = curr_frm.id_;
//                 lm = nullptr;
//                 continue;
//             }
//
//             ++num_valid_matches;
//         }
//
//     }
//
//     return num_valid_matches;
// }

} // namespace module
} // namespace openvslam
