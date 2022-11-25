#include "openvslam/config.h"
#include "openvslam/system.h"
#include "openvslam/tracking_module.h"
#include "openvslam/mapping_module.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/multi_frame.h"
#include "openvslam/data/map_database.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/projection.h"
#include "openvslam/module/local_map_updater.h"
#include "openvslam/util/image_converter.h"
#include "openvslam/util/yaml.h"

#include <chrono>
#include <unordered_map>

#include <spdlog/spdlog.h>

using namespace std;

namespace {
using namespace openvslam;

feature::orb_params get_orb_params(const YAML::Node& yaml_node) {
    spdlog::debug("load ORB parameters");
    try {
        return feature::orb_params(yaml_node);
    }
    catch (const std::exception& e) {
        spdlog::error("failed in loading ORB parameters: {}", e.what());
        throw;
    }
}

double get_true_depth_thr(const camera::base* camera, const YAML::Node& yaml_node) {
    spdlog::debug("load depth threshold");
    double true_depth_thr = 40.0;
    if (camera->setup_type_ == camera::setup_type_t::Stereo || camera->setup_type_ == camera::setup_type_t::RGBD) {
        const auto depth_thr_factor = yaml_node["depth_threshold"].as<double>(40.0);
        true_depth_thr = camera->true_baseline_ * depth_thr_factor;
    }
    return true_depth_thr;
}

double get_depthmap_factor(const camera::base* camera, const YAML::Node& yaml_node) {
    spdlog::debug("load depthmap factor");
    double depthmap_factor = 1.0;
    if (camera->setup_type_ == camera::setup_type_t::RGBD) {
        depthmap_factor = yaml_node["depthmap_factor"].as<double>(depthmap_factor);
    }
    if (depthmap_factor < 0.) {
        throw std::runtime_error("depthmap_factor must be greater than 0");
    }
    return depthmap_factor;
}

} // unnamed namespace

namespace openvslam {

tracking_module::tracking_module(const std::shared_ptr<config>& cfg, system* system, data::map_database* map_db,
                                 data::bow_vocabulary* bow_vocab)
    : cam_rig_(cfg->cam_rig_.get()),
      // todo ivan. fix using cam_rig_->cameras[0]
      true_depth_thr_(get_true_depth_thr(cam_rig_->cameras[0].get(), util::yaml_optional_ref(cfg->yaml_node_, "Tracking"))),
      depthmap_factor_(get_depthmap_factor(cam_rig_->cameras[0].get(), util::yaml_optional_ref(cfg->yaml_node_, "Tracking"))),

      system_(system), map_db_(map_db), bow_vocab_(bow_vocab),
      initializer_(cfg->cam_rig_->isMono(), map_db, util::yaml_optional_ref(cfg->yaml_node_, "Initializer")),
      frame_tracker_(10),
      pose_optimizer_(),
      keyfrm_inserter_(cfg->cam_rig_->isMono(), true_depth_thr_, map_db, 0, cfg->cam_rig_->cameras[0]->fps_) {
    spdlog::debug("CONSTRUCT: tracking_module");

    feature::orb_params orb_params = get_orb_params(util::yaml_optional_ref(cfg->yaml_node_, "Feature"));
    const auto tracking_params = util::yaml_optional_ref(cfg->yaml_node_, "Tracking");
    const auto max_num_keypoints = tracking_params["max_num_keypoints"].as<unsigned int>(2000);
    extractor_left_ = new feature::orb_extractor(max_num_keypoints, orb_params);
    if (cam_rig_->isMono()) {
        const auto ini_max_num_keypoints = tracking_params["ini_max_num_keypoints"].as<unsigned int>(2 * extractor_left_->get_max_num_keypoints());
        ini_extractor_left_ = new feature::orb_extractor(ini_max_num_keypoints, orb_params);
    }
    // todo ivan. fix this multi cam
    if (cam_rig_->cameras.size()==2) {
        extractor_right_ = new feature::orb_extractor(max_num_keypoints, orb_params);
    }
}

tracking_module::~tracking_module() {
    delete extractor_left_;
    extractor_left_ = nullptr;
    delete extractor_right_;
    extractor_right_ = nullptr;
    delete ini_extractor_left_;
    ini_extractor_left_ = nullptr;

    spdlog::debug("DESTRUCT: tracking_module");
}

void tracking_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    keyfrm_inserter_.set_mapping_module(mapper);
}

bool tracking_module::get_mapping_module_status() const {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    return mapping_is_enabled_;
}

std::vector<cv::KeyPoint> tracking_module::get_initial_keypoints() const {
    return initializer_.get_initial_keypoints();
}

std::vector<int> tracking_module::get_initial_matches() const {
    return initializer_.get_initial_matches();
}


std::shared_ptr<Mat44_t> tracking_module::track_multi_images(const vector<cv::Mat>& imgs,
                                                             const double timestamp, const vector<cv::Mat>& masks) {
    AssertLog(imgs.size()<=2, "limit to two for now");

    // if (imgs.size()==1){
    //     return track_monocular_image(imgs[0], timestamp, masks.empty()? cv::Mat(): masks[0]);
    // } else{
    //     return track_stereo_image(imgs[0], imgs[1], timestamp, masks.empty()? cv::Mat(): masks[0]);
    // }

    vector<cv::Mat> empty_masks;
    if (masks.empty()) empty_masks.resize(imgs.size());
    const vector<cv::Mat> &final_masks = masks.empty()? empty_masks : masks;

    imgs_gray_ = imgs;
    curr_frm_ = data::MultiFrame(cam_rig_);

    for (int i = 0; i < imgs.size(); ++i) {
        util::convert_to_grayscale(imgs_gray_[i], cam_rig_->cameras[i]->color_order_);
        curr_frm_.frames.push_back(shared_ptr<data::frame>(new data::frame(
            imgs_gray_[i], timestamp, extractor_left_, bow_vocab_,
            cam_rig_->cameras[i].get(), true_depth_thr_, final_masks[i])));
    }

    track();

    std::shared_ptr<Mat44_t> cam_pose_wc = nullptr;
    if (curr_frm_.cam_pose_cw_is_valid_) {
        cam_pose_wc = std::allocate_shared<Mat44_t>(Eigen::aligned_allocator<Mat44_t>(), curr_frm_.get_cam_pose_inv());
    }
    return cam_pose_wc;
}

void tracking_module::reset() {
    spdlog::info("resetting system");

    initializer_.reset();
    keyfrm_inserter_.reset();

    mapper_->request_reset();

    map_db_->clear();

    data::frame::next_id_ = 0;
    data::keyframe::next_id_ = 0;
    data::landmark::next_id_ = 0;
    data::MultiFrame::next_id_ = 0;
    data::MultiKeyframe::next_id_ = 0;

    last_reloc_frm_id_ = 0;

    tracking_state_ = tracker_state_t::NotInitialized;
}

void tracking_module::track() {
    if (tracking_state_ == tracker_state_t::NotInitialized) {
        tracking_state_ = tracker_state_t::Initializing;
    }

    last_tracking_state_ = tracking_state_;

    // check if pause is requested
    check_and_execute_pause();
    while (is_paused()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // LOCK the map database
    // todo [ivan] man, this is a long lock...
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    if (tracking_state_ == tracker_state_t::Initializing) {
        if (!initialize()) {
            return;
        }

        // update the reference keyframe, local keyframes, and local landmarks
        update_local_map();

        // pass all of the keyframes to the mapping module
        const auto keyfrms = map_db_->get_all_keyframes();
        for (const auto& keyfrm : keyfrms) {
            mapper_->queue_keyframe(keyfrm);
        }

        // state transition to Tracking mode
        tracking_state_ = tracker_state_t::Tracking;
    }
    else{
        // [ivan] very bad logic for landmark replacement. need this as fuse could have happened,
        //  and landmarks only keep knowledge of keyframe observations, not for this normal frame
        // apply replace of landmarks observed in the last frame
        apply_landmark_replace();
        // update the camera pose of the last frame
        // because the mapping module might optimize the camera pose of the last frame's reference keyframe
        update_last_frame();

        // set the reference keyframe of the current frame
        curr_frm_.ref_keyfrm_ = last_frm_.ref_keyfrm_;

        auto succeeded = track_current_frame();

        // update the local map and optimize the camera pose of the current frame
        if (succeeded) {
            update_local_map();
            succeeded = optimize_current_frame_with_local_map();
        }

        // update the motion model
        if (succeeded) {
            update_motion_model();
        }

        // state transition
        tracking_state_ = succeeded ? tracker_state_t::Tracking : tracker_state_t::Lost;

        // show message if tracking has been lost
        if (last_tracking_state_ != tracker_state_t::Lost && tracking_state_ == tracker_state_t::Lost) {
            spdlog::info("tracking lost: frame {}", curr_frm_.id_);
        }

        // check to insert the new keyframe derived from the current frame
        if (succeeded && new_keyframe_is_needed()) {
            insert_new_keyframe();
        }
    }

    // store the relative pose from the reference keyframe to the current frame
    // to update the camera pose at the beginning of the next tracking process
    if (curr_frm_.cam_pose_cw_is_valid_) {
        last_cam_pose_from_ref_keyfrm_ = curr_frm_.cam_pose_cw_ * curr_frm_.ref_keyfrm_->getCamPoseInv();
    }

    // update last frame
    last_frm_ = curr_frm_;
}


bool tracking_module::initialize() {
    // try to initialize with the current frame
    initializer_.initialize(curr_frm_);

    // if map building was failed -> reset the map database
    if (initializer_.get_state() == module::initializer_state_t::Wrong) {
        // reset
        system_->request_reset();
        return false;
    }

    // if initializing was failed -> try to initialize with the next frame
    if (initializer_.get_state() != module::initializer_state_t::Succeeded) {
        return false;
    }

    // succeeded
    return true;
}

bool tracking_module::track_current_frame() {
    bool succeeded = false;

    if (tracking_state_ == tracker_state_t::Tracking) {
        // Tracking mode
        if (twist_is_valid_ && last_reloc_frm_id_ + 2 < curr_frm_.id_) {
            // if the motion model is valid
            spdlog::debug("track with motion");
            succeeded = frame_tracker_.motion_based_track(curr_frm_, last_frm_, twist_);
        }
        // if (!succeeded) {
        //     succeeded = frame_tracker_.bow_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
        // }
        if (!succeeded) {
            spdlog::debug("track with robust");
            succeeded = frame_tracker_.robust_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
        }
    }

    return succeeded;
}

void tracking_module::update_motion_model() {
    if (last_frm_.cam_pose_cw_is_valid_) {
        Mat44_t last_frm_cam_pose_wc = Mat44_t::Identity();
        last_frm_cam_pose_wc.block<3, 3>(0, 0) = last_frm_.get_rotation_inv();
        last_frm_cam_pose_wc.block<3, 1>(0, 3) = last_frm_.get_cam_center();
        twist_is_valid_ = true;
        twist_ = curr_frm_.cam_pose_cw_ * last_frm_cam_pose_wc;
    }
    else {
        twist_is_valid_ = false;
        twist_ = Mat44_t::Identity();
    }
}

void tracking_module::apply_landmark_replace() {
    for (auto &frm : last_frm_.frames) {
        for (unsigned int idx = 0; idx < frm->num_keypts_; ++idx) {
            auto& lm = frm->landmarks_.at(idx);
            if (!lm) {
                continue;
            }

            // note ivan. after this operation, lm should get destroyed as no-one owns it
            std::shared_ptr<data::landmark> replaced_lm = lm->get_replaced();
            if (replaced_lm) {
                frm->landmarks_.at(idx) = replaced_lm;
            }
        }
    }
}

void tracking_module::update_last_frame() {
    auto last_ref_keyfrm = last_frm_.ref_keyfrm_;
    if (!last_ref_keyfrm) {
        return;
    }
    last_frm_.set_cam_pose(last_cam_pose_from_ref_keyfrm_ * last_ref_keyfrm->getCamPose());
}

bool tracking_module::optimize_current_frame_with_local_map() {
    // acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
    search_local_landmarks();

    // optimize the pose
    num_tracked_lms_ = pose_optimizer_.optimize(curr_frm_);

    // [ivan] this is the final frame optimization, we call increase_num_observed
    for (int i = 0; i < curr_frm_.size(); ++i) {
        for (auto &lm : curr_frm_[i]->landmarks_) {
            if (lm && !lm->will_be_erased()) lm->increase_num_observed();
        }
    }

    constexpr unsigned int num_tracked_lms_thr = 20;

    // if recently relocalized, use the more strict threshold
    if (curr_frm_.id_ < last_reloc_frm_id_ + cam_rig_->cameras[0]->fps_ && num_tracked_lms_ < 2 * num_tracked_lms_thr) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, 2 * num_tracked_lms_thr);
        return false;
    }

    // check the threshold of the number of tracked landmarks
    if (num_tracked_lms_ < num_tracked_lms_thr) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, num_tracked_lms_thr);
        return false;
    }

    return true;
}

void tracking_module::update_local_map() {
    // clean landmark associations
    // todo ivan. we'll just do it again in search_local_landmarks.....
    for(auto &frm : curr_frm_.frames){
        for (unsigned int idx = 0; idx < frm->num_keypts_; ++idx) {
            const auto& lm = frm->landmarks_.at(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                frm->landmarks_.at(idx) = nullptr;
                continue;
            }
        }
    }

    // acquire the current local map
    constexpr unsigned int max_num_local_keyfrms = 60;

    local_landmarks_ = module::local_map_updater::GetLocalLandmarks(curr_frm_, map_db_, max_num_local_keyfrms);

    map_db_->set_local_landmarks(local_landmarks_);
}

void tracking_module::search_local_landmarks() {
    // select the landmarks which can be reprojected from the ones observed in the current frame
    for(auto &frm : curr_frm_.frames) {
        for (const auto& lm : frm->landmarks_) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // this landmark cannot be reprojected
            // because already observed in the current frame
            // todo ivan. this is just stupid, previously we set outlier lm is_observable_in_tracking_ = false
            // lm->is_observable_in_tracking_ = false;

            // // todo [ivan] doing here. should kept for each frame, not the multi frame
            lm->identifier_in_local_lm_search_ = curr_frm_.id_;

            // this landmark is observable from the current frame
            lm->increase_num_observable();
        }
    }

    vector<vector<match::ObservableLandmark>> observable_lms(curr_frm_.frames.size());
    for (auto& lm : local_landmarks_) {
        // avoid the landmarks which cannot be reprojected (== observed in the current frame)
        if (lm->identifier_in_local_lm_search_ == curr_frm_.id_) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // check the observability
        for (int i=0; i<curr_frm_.frames.size(); ++i){
            observable_lms.at(i).reserve(local_landmarks_.size());
            Vec2_t reproj;
            float x_right;
            unsigned int pred_scale_level;
            if (curr_frm_.frames[i]->can_observe(lm, 0.5, reproj, x_right, pred_scale_level)){
                observable_lms.at(i).push_back({lm, reproj.x(), reproj.y(), x_right, pred_scale_level});
                lm->increase_num_observable();
            }
        }
    }

    // acquire more 2D-3D matches by projecting the local landmarks to the current frame
    match::projection projection_matcher(0.8);
    const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
                             ? 20.0 : 10;
    for(int i=0; i<curr_frm_.frames.size(); ++i){
        projection_matcher.match_frame_and_landmarks(*curr_frm_.frames[i], observable_lms[i], margin);
    }
}

bool tracking_module::new_keyframe_is_needed() const {
    // cannnot insert the new keyframe in a second after relocalization
    const auto num_keyfrms = map_db_->get_num_keyframes();
    if (cam_rig_->cameras[0]->fps_ < num_keyfrms && curr_frm_.id_ < last_reloc_frm_id_ + cam_rig_->cameras[0]->fps_) {
        return false;
    }

    // check the new keyframe is needed
    return keyfrm_inserter_.new_keyframe_is_needed(curr_frm_, num_tracked_lms_, *curr_frm_.ref_keyfrm_);
}

void tracking_module::insert_new_keyframe() {
    // insert the new keyframe
    const auto ref_keyfrm = keyfrm_inserter_.insert_new_keyframe(curr_frm_);
    // set the reference keyframe with the new keyframe
    if (ref_keyfrm) {
        curr_frm_.ref_keyfrm_ = ref_keyfrm;
    }
}

void tracking_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

bool tracking_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool tracking_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void tracking_module::resume() {
    std::lock_guard<std::mutex> lock(mtx_pause_);

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume tracking module");
}

bool tracking_module::check_and_execute_pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    if (pause_is_requested_) {
        is_paused_ = true;
        spdlog::info("pause tracking module");
        return true;
    }
    else {
        return false;
    }
}

} // namespace openvslam
