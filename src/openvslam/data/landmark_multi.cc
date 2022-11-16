#include "openvslam/data/frame.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/match/base.h"

#include <spdlog/spdlog.h>

using namespace std;

namespace openvslam {
namespace data {

void ObservationBook::add(const shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx, unsigned int kp_idx) {
    AssertLog(frm_idx>=0 && frm_idx<keyfrm->frames.size(), "");

    if (keyfrm->frames[frm_idx]->stereo_x_right_.at(kp_idx)>=0){
        num_observations += 2;
    } else {
        num_observations += 1;
    }

    auto itr = observations.find(keyfrm->id_);
    if (itr==observations.end()){
        observations.insert({keyfrm->id_, {keyfrm, {{frm_idx, kp_idx}}} });
    } else {
        Observation &observation = itr->second;
        AssertLog(!observation.frm_kp.empty(), "exists empty observation");
        AssertLog(observation.frm_kp.find(frm_idx) == observation.frm_kp.end(), "already have this observation");
        observation.frm_kp[frm_idx] = kp_idx;
    }
}

void ObservationBook::erase(const shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx) {
    AssertLog(frm_idx>=0 && frm_idx<keyfrm->frames.size(), "");
    auto itr = observations.find(keyfrm->id_);

    if (itr == observations.end()){
        spdlog::warn("landmark erase could not find the observation of MultiKeyframe");
        return ;
    }

    Observation &observation = itr->second;
    AssertLog(!observation.frm_kp.empty(), "exists empty observation");

    auto itr2 = observation.frm_kp.find(frm_idx);
    if (itr2 == observation.frm_kp.end()){
        spdlog::warn("landmark erase could not find the observation of the keyframe");
        return ;
    }

    if (0 <= keyfrm->frames[frm_idx]->stereo_x_right_.at(itr2->second)) {
        num_observations -= 2;
    }
    else {
        num_observations -= 1;
    }

    if (observation.frm_kp.size()==1){
        observations.erase(keyfrm->id_);
    } else{
        observation.frm_kp.erase(itr2);
    }
}

std::atomic<unsigned int> landmark::next_id_{0};

landmark::landmark(const Vec3_t& pos_w, const Observation& ref_obs, map_database* map_db)
    : id_(next_id_++), pos_w_(pos_w), ref_obs_(ref_obs), map_db_(map_db) {
    AssertLog(ref_obs.frm_kp.size()==1, "");
}

void landmark::set_pos_in_world(const Vec3_t& pos_w) {
    std::lock_guard<std::mutex> lock(mtx_position_);
    pos_w_ = pos_w;
}

Vec3_t landmark::get_pos_in_world() const {
    std::lock_guard<std::mutex> lock(mtx_position_);
    return pos_w_;
}

Vec3_t landmark::get_obs_mean_normal() const {
    std::lock_guard<std::mutex> lock(mtx_position_);
    return mean_normal_;
}

void landmark::add_observation(const shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx, unsigned int kp_idx) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    observations_.add(keyfrm, frm_idx, kp_idx);
}

void landmark::erase_observation(const shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx) {
    bool discard = false;
    {
        std::lock_guard<std::mutex> lock(mtx_observations_);
        observations_.erase(keyfrm, frm_idx);

        // If only 2 observations or less, discard point
        if (observations_.num_observations <= 2) {
            discard = true;
        }
        // replace reference if needed
        else if (ref_obs_.keyfrm.lock() == keyfrm){
            unsigned int kf_id = observations_.observations.begin()->first;
            for (const auto &pair : observations_.observations) {
                if (pair.first< kf_id){
                    ref_obs_ = pair.second;
                }
            }
        }
    }

    if (discard) {
        prepare_for_erasing();
    }
}

ObservationBook landmark::get_observations() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return observations_;
}

unsigned int landmark::num_observations() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return observations_.num_observations;
}

bool landmark::has_observation() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return 0 < observations_.num_observations;
}

bool landmark::is_observed_in_keyframe(const std::shared_ptr<keyframe>& keyfrm) const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    auto itr1 = observations_.observations.find(keyfrm->parent_->id_);
    if (itr1 == observations_.observations.end()) return false;

    auto itr2 = itr1->second.frm_kp.find(keyfrm->sibling_idx_);
    return itr2 != itr1->second.frm_kp.end();
}

cv::Mat landmark::get_descriptor() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return descriptor_.clone();
}

void landmark::compute_descriptor() {
    ObservationBook observations;
    {
        std::lock_guard<std::mutex> lock1(mtx_observations_);
        if (will_be_erased_) {
            return;
        }
        observations = observations_;
    }

    if (observations.observations.empty()) {
        return;
    }

    // Append features of corresponding points
    std::vector<cv::Mat> descriptors;
    size_t n =0;
    for (const auto& obs : observations.observations) {
        n += obs.second.frm_kp.size();
    }
    descriptors.reserve(n);
    for (const auto& obs : observations.observations) {
        auto keyfrm = obs.second.keyfrm.lock();
        AssertLog(keyfrm, "keyframe not exists");
        for (const auto pair : obs.second.frm_kp) {
            descriptors.push_back(keyfrm->frames[pair.first]->descriptors_.row(pair.second));
        }
    }

    // Get median of Hamming distance
    // Calculate all the Hamming distances between every pair of the features
    const auto num_descs = descriptors.size();
    std::vector<std::vector<unsigned int>> hamm_dists(num_descs, std::vector<unsigned int>(num_descs));
    for (unsigned int i = 0; i < num_descs; ++i) {
        hamm_dists.at(i).at(i) = 0;
        for (unsigned int j = i + 1; j < num_descs; ++j) {
            const auto dist = match::compute_descriptor_distance_32(descriptors.at(i), descriptors.at(j));
            hamm_dists.at(i).at(j) = dist;
            hamm_dists.at(j).at(i) = dist;
        }
    }

    // Get the nearest value to median
    unsigned int best_median_dist = match::MAX_HAMMING_DIST;
    unsigned int best_idx = 0;
    for (unsigned idx = 0; idx < num_descs; ++idx) {
        std::vector<unsigned int> partial_hamm_dists(hamm_dists.at(idx).begin(), hamm_dists.at(idx).begin() + num_descs);
        std::sort(partial_hamm_dists.begin(), partial_hamm_dists.end());
        const auto median_dist = partial_hamm_dists.at(static_cast<unsigned int>(0.5 * (num_descs - 1)));

        if (median_dist < best_median_dist) {
            best_median_dist = median_dist;
            best_idx = idx;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx_observations_);
        descriptor_ = descriptors.at(best_idx).clone();
    }
}

void landmark::update_normal_and_depth() {
    ObservationBook observations;
    std::shared_ptr<MultiKeyframe> ref_keyfrm = nullptr;
    std::shared_ptr<keyframe> ref_single_kf = nullptr;
    unsigned int ref_kp_ind;
    Vec3_t pos_w;
    {
        std::scoped_lock lock(mtx_observations_, mtx_position_);
        if (will_be_erased_) {
            return;
        }
        observations = observations_;
        ref_keyfrm = ref_obs_.keyfrm.lock();
        AssertLog(ref_keyfrm, "keyframe not exists");
        ref_single_kf = ref_keyfrm->frames[ref_obs_.frm_kp.begin()->first];
        ref_kp_ind = ref_obs_.frm_kp.begin()->second;
        pos_w = pos_w_;
    }

    if (observations.observations.empty()) {
        return;
    }

    Vec3_t mean_normal = Vec3_t::Zero();
    unsigned int num_observations = 0;
    for (const auto& obs : observations.observations) {
        auto keyfrm = obs.second.keyfrm.lock();
        AssertLog(keyfrm, "keyframe not exists");
        for (const auto pair : obs.second.frm_kp) {
            const Vec3_t cam_center = keyfrm->frames[pair.first]->get_cam_center();
            const Vec3_t normal = pos_w - cam_center;
            mean_normal = mean_normal + normal.normalized();
            ++num_observations;
        }
    }

    const Vec3_t cam_to_lm_vec = pos_w - ref_single_kf->get_cam_center();
    const auto dist = cam_to_lm_vec.norm();
    const auto scale_level = ref_single_kf->undist_keypts_.at(ref_kp_ind).octave;
    const auto scale_factor = ref_single_kf->scale_factors_.at(ref_kp_ind);
    const auto num_scale_levels = ref_single_kf->num_scale_levels_;

    {
        std::lock_guard<std::mutex> lock3(mtx_position_);
        max_valid_dist_ = dist * scale_factor;
        min_valid_dist_ = max_valid_dist_ / ref_single_kf->scale_factors_.at(num_scale_levels - 1);
        mean_normal_ = mean_normal.normalized();
    }
}

float landmark::get_min_valid_distance() const {
    std::lock_guard<std::mutex> lock(mtx_position_);
    return 0.7 * min_valid_dist_;
}

float landmark::get_max_valid_distance() const {
    std::lock_guard<std::mutex> lock(mtx_position_);
    return 1.3 * max_valid_dist_;
}

unsigned int landmark::predict_scale_level(const float cam_to_lm_dist, const frame* frm) const {
    float ratio;
    {
        std::lock_guard<std::mutex> lock(mtx_position_);
        ratio = max_valid_dist_ / cam_to_lm_dist;
    }

    const auto pred_scale_level = static_cast<int>(std::ceil(std::log(ratio) / frm->log_scale_factor_));
    if (pred_scale_level < 0) {
        return 0;
    }
    else if (frm->num_scale_levels_ <= static_cast<unsigned int>(pred_scale_level)) {
        return frm->num_scale_levels_ - 1;
    }
    else {
        return static_cast<unsigned int>(pred_scale_level);
    }
}

unsigned int landmark::predict_scale_level(const float cam_to_lm_dist, const std::shared_ptr<keyframe>& keyfrm) const {
    float ratio;
    {
        std::lock_guard<std::mutex> lock(mtx_position_);
        ratio = max_valid_dist_ / cam_to_lm_dist;
    }

    const auto pred_scale_level = static_cast<int>(std::ceil(std::log(ratio) / keyfrm->log_scale_factor_));
    if (pred_scale_level < 0) {
        return 0;
    }
    else if (keyfrm->num_scale_levels_ <= static_cast<unsigned int>(pred_scale_level)) {
        return keyfrm->num_scale_levels_ - 1;
    }
    else {
        return static_cast<unsigned int>(pred_scale_level);
    }
}

// note ivan. why prepare in naming? this essentially calls all owners to release itself, after that only local copy exists
void landmark::prepare_for_erasing() {
    ObservationBook observations;
    {
        std::scoped_lock lock(mtx_observations_, mtx_position_);
        observations = observations_;
        observations_.observations.clear();
        observations_.num_observations = 0;
        will_be_erased_ = true;
    }

    for (const auto& obs : observations.observations) {
        auto keyfrm = obs.second.keyfrm.lock();
        AssertLog(keyfrm, "keyframe not exists");
        for (auto pair : obs.second.frm_kp) {
            keyfrm->frames[pair.first]->erase_landmark_with_index(pair.second);
        }
    }

    map_db_->erase_landmark(this->id_);
}

bool landmark::will_be_erased() {
    std::lock_guard<std::mutex> lock1(mtx_observations_);
    std::lock_guard<std::mutex> lock2(mtx_position_);
    return will_be_erased_;
}

void landmark::replace(std::shared_ptr<landmark> lm) {
    if (lm->id_ == this->id_) {
        return;
    }

    unsigned int num_observable, num_observed;
    ObservationBook observations;
    {
        std::scoped_lock lock(mtx_observations_, mtx_position_);
        observations = observations_;
        observations_.observations.clear();
        observations_.num_observations = 0;
        will_be_erased_ = true;
        num_observable = num_observable_;
        num_observed = num_observed_;
        replaced_ = lm;
    }

    for (const auto &obs : observations.observations){
        auto keyfrm = obs.second.keyfrm.lock();
        AssertLog(keyfrm, "keyframe not exists");
        for (const auto& pair : obs.second.frm_kp) {

            // todo ivan. check the logic, should lm get this landmark's observation as well?
            if (!lm->is_observed_in_keyframe(keyfrm)) {
                keyfrm->replace_landmark(lm, keyfrm_and_idx.second);
                lm->add_observation(keyfrm, keyfrm_and_idx.second);
            }
            else {
                keyfrm->erase_landmark_with_index(keyfrm_and_idx.second);
            }

            // todo ivan. doing here
        }
    }

    lm->increase_num_observed(num_observed);
    lm->increase_num_observable(num_observable);
    lm->compute_descriptor();

    map_db_->erase_landmark(this->id_);
}

std::shared_ptr<landmark> landmark::get_replaced() const {
    std::scoped_lock lock(mtx_observations_, mtx_position_);
    return replaced_;
}

void landmark::increase_num_observable(unsigned int num_observable) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    num_observable_ += num_observable;
}

void landmark::increase_num_observed(unsigned int num_observed) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    num_observed_ += num_observed;
}

float landmark::get_observed_ratio() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return static_cast<float>(num_observed_) / num_observable_;
}


} // namespace data
} // namespace openvslam
