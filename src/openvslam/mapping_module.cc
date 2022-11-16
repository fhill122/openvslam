#include "openvslam/type.h"
#include "openvslam/mapping_module.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/match/fuse.h"
#include "openvslam/match/robust.h"
#include "openvslam/module/two_view_triangulator.h"
#include "openvslam/solve/essential_solver.h"

#include <unordered_set>
#include <thread>

#include <spdlog/spdlog.h>

using namespace std;

namespace openvslam {

mapping_module::mapping_module(const YAML::Node& yaml_node, data::map_database* map_db)
    : local_map_cleaner_(new module::local_map_cleaner(yaml_node["redundant_obs_ratio_thr"].as<double>(0.9))), map_db_(map_db),
      local_bundle_adjuster_(new optimize::local_bundle_adjuster()) {
    spdlog::debug("CONSTRUCT: mapping_module");
    spdlog::debug("load mapping parameters");

    spdlog::debug("load monocular mappping parameters");
    if (yaml_node["baseline_dist_thr"]) {
        if (yaml_node["baseline_dist_thr_ratio"]) {
            throw std::runtime_error("Do not set both baseline_dist_thr_ratio and baseline_dist_thr.");
        }
        baseline_dist_thr_ = yaml_node["baseline_dist_thr"].as<double>(1.0);
        use_baseline_dist_thr_ratio_ = false;
        spdlog::debug("Use baseline_dist_thr: {}", baseline_dist_thr_);
    }
    else {
        baseline_dist_thr_ratio_ = yaml_node["baseline_dist_thr_ratio"].as<double>(0.02);
        use_baseline_dist_thr_ratio_ = true;
        spdlog::debug("Use baseline_dist_thr_ratio: {}", baseline_dist_thr_ratio_);
    }
}

mapping_module::~mapping_module() {
    spdlog::debug("DESTRUCT: mapping_module");
}

void mapping_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}


void mapping_module::run() {
    spdlog::info("start mapping module");

    is_terminated_ = false;

    while (true) {
        // waiting time for the other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // LOCK
        set_keyframe_acceptability(false);

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if pause is requested
        if (pause_is_requested()) {
            // if any keyframe is queued, all of them must be processed before the pause
            while (keyframe_is_queued()) {
                // create and extend the map with the new keyframe
                mapping_with_new_keyframe();
            }
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset, UNLOCK and continue
            reset();
            set_keyframe_acceptability(true);
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            // UNLOCK and continue
            set_keyframe_acceptability(true);
            continue;
        }

        // create and extend the map with the new keyframe
        mapping_with_new_keyframe();

        // LOCK end
        set_keyframe_acceptability(true);
    }

    spdlog::info("terminate mapping module");
}

void mapping_module::queue_keyframe(const std::shared_ptr<data::MultiKeyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    keyfrms_queue_.push_back(keyfrm);
    abort_local_BA_ = true;
}

unsigned int mapping_module::get_num_queued_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return keyfrms_queue_.size();
}

bool mapping_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

bool mapping_module::get_keyframe_acceptability() const {
    return keyfrm_acceptability_;
}

void mapping_module::set_keyframe_acceptability(const bool acceptability) {
    keyfrm_acceptability_ = acceptability;
}

void mapping_module::abort_local_BA() {
    abort_local_BA_ = true;
}

void mapping_module::mapping_with_new_keyframe() {
    // dequeue
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        // dequeue -> cur_keyfrm_
        cur_keyfrm_ = keyfrms_queue_.front();
        keyfrms_queue_.pop_front();
    }

    // store the new keyframe to the database
    store_new_keyframe();

    // remove redundant landmarks
    local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_, false);

    // todo [ivan] this logic is not proper:
    //  currently do at least 1 triangulation, then new kf can interrupt both triangulation and ba.
    //  if triangulation takes too long, ba never got chance to run
    //  should triangulate at least n keyframes, do optimization m times (if still needed), repeat...

    // triangulate new landmarks between the current frame and each of the covisibilities
    create_new_landmarks();

    if (keyframe_is_queued()) {
        return;
    }

    // todo [ivan] bench
    // detect and resolve the duplication of the landmarks observed in the current frame
    update_new_keyframe();

    if (keyframe_is_queued() || pause_is_requested()) {
        return;
    }

    // local bundle adjustment
    abort_local_BA_ = false;
    if (2 < map_db_->get_num_keyframes()) {
        local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_);
    }
}

void mapping_module::store_new_keyframe() {
    // compute BoW feature vector
    for(auto &kf : cur_keyfrm_->frames) {
        kf->compute_bow();

        // update graph
        const auto cur_lms = kf->get_landmarks();
        for (unsigned int idx = 0; idx < cur_lms.size(); ++idx) {
            auto lm = cur_lms.at(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // todo [ivan] check this. keyframe just created, how come lm would ever have observation of curr_keyfrm_?!
            // if `lm` does not have the observation information from `cur_keyfrm_`,
            // add the association between the keyframe and the landmark
            if (lm->is_observed_in_keyframe(kf)) {
                AssertLog(false, "checking, this should never happen");
                // if `lm` is correctly observed, make it be checked by the local map cleaner
                local_map_cleaner_->add_fresh_landmark(lm);
                continue;
            }

            // update connection
            lm->add_observation(kf, idx);
            // update geometry
            lm->update_normal_and_depth();
            lm->compute_descriptor();
        }
    }

    // store the new keyframe to the map database
    map_db_->add_keyframe(cur_keyfrm_);
}

void mapping_module::create_new_landmarks() {
    constexpr int kNumNeighbor = 20;
    // lowe's_ratio will not be used
    match::robust robust_matcher(0.0, false);


    for (unsigned int i = 0; i < kNumNeighbor; ++i) {
        // if any keyframe is queued and we have done 1 triangulation, abort
        if (1 < i && keyframe_is_queued()) {
            return;
        }

        // get the neighbor keyframe
        auto ngh_keyfrm = map_db_->getKeyframe(cur_keyfrm_->id_-i);
        if (!ngh_keyfrm) return ;

        // todo [ivan] should we allow cross camera triangulation? not doing here
        for (int j=0; j<cur_keyfrm_->frames.size(); ++j){
            // camera center of the current keyframe
            const Vec3_t cur_cam_center = cur_keyfrm_->at(j)->get_cam_center();
            // camera center of the neighbor keyframe
            const Vec3_t ngh_cam_center = ngh_keyfrm->at(j)->get_cam_center();

            // compute the baseline between the current and neighbor keyframes
            const Vec3_t baseline_vec = ngh_cam_center - cur_cam_center;
            const auto baseline_dist = baseline_vec.norm();

            // if the scene scale is much smaller than the baseline, abort the triangulation
            if (use_baseline_dist_thr_ratio_) {
                const float median_depth_in_ngh = ngh_keyfrm->frames[j]->compute_median_depth(true);
                if (baseline_dist < baseline_dist_thr_ratio_ * median_depth_in_ngh) {
                    continue;
                }
            }
            else {
                if (baseline_dist < baseline_dist_thr_) {
                    continue;
                }
            }

            // estimate matches between the current and neighbor keyframes,
            // then reject outliers using Essential matrix computed from the two camera poses

            // (cur bearing) * E_ngh_to_cur * (ngh bearing) = 0
            // const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm, cur_keyfrm_);
            const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(
                                    ngh_keyfrm->at(j)->get_rotation(), ngh_keyfrm->at(j)->get_translation(),
                                    cur_keyfrm_->at(j)->get_rotation(), cur_keyfrm_->at(j)->get_translation());

            // vector of matches (idx in the current, idx in the neighbor)
            std::vector<std::pair<unsigned int, unsigned int>> matches;
            robust_matcher.match_for_triangulation(cur_keyfrm_->at(j), ngh_keyfrm->at(j), E_ngh_to_cur, matches);

            // triangulation
            triangulate_with_two_keyframes(cur_keyfrm_->at(j), ngh_keyfrm->at(j), matches);
        }
    }
}

void mapping_module::triangulate_with_two_keyframes(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2,
                                                    const std::vector<std::pair<unsigned int, unsigned int>>& matches) {
    const module::two_view_triangulator triangulator(keyfrm_1, keyfrm_2, 1.0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < matches.size(); ++i) {
        const auto idx_1 = matches.at(i).first;
        const auto idx_2 = matches.at(i).second;

        // triangulate between idx_1 and idx_2
        Vec3_t pos_w;
        if (!triangulator.triangulate(idx_1, idx_2, pos_w)) {
            // failed
            continue;
        }
        // succeeded

        // create a landmark object
        auto lm = std::make_shared<data::landmark>(pos_w, keyfrm_1, map_db_);

        lm->add_observation(keyfrm_1, idx_1);
        lm->add_observation(keyfrm_2, idx_2);

        keyfrm_1->add_landmark(lm, idx_1);
        keyfrm_2->add_landmark(lm, idx_2);

        lm->compute_descriptor();
        lm->update_normal_and_depth();

        map_db_->add_landmark(lm);
        // wait for redundancy check
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
            local_map_cleaner_->add_fresh_landmark(lm);
        }
    }
}

void mapping_module::update_new_keyframe() {
    // get the targets to check landmark fusion
    const auto fuse_tgt_mkeyfrms = map_db_->getKeyframes<unordered_set>(cur_keyfrm_->id_-10, cur_keyfrm_->id_);
    // todo [ivan] use raw pointer
    vector<shared_ptr<data::keyframe>> fuse_tgt_keyfrms;
    fuse_tgt_keyfrms.reserve(fuse_tgt_mkeyfrms.size() * cur_keyfrm_->size());
    for (auto &mkf : fuse_tgt_mkeyfrms){
        fuse_tgt_keyfrms.insert(fuse_tgt_keyfrms.end(), mkf->frames.begin(), mkf->frames.end());
    }

    for (const auto &kf : cur_keyfrm_->frames){
        // resolve the duplication of landmarks between the current keyframe and the targets
        fuse_landmark_duplication(kf, fuse_tgt_keyfrms);

        // update the geometries
        const auto cur_landmarks = kf->get_landmarks();
        for (const auto& lm : cur_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            lm->compute_descriptor();
            lm->update_normal_and_depth();
        }
    }
}

template < template<typename , typename...> typename Container, typename... ContainerParams>
void mapping_module::fuse_landmark_duplication(const std::shared_ptr<data::keyframe> &cur_keyfrm,
                   const Container<std::shared_ptr<data::keyframe>, ContainerParams...>& fuse_tgt_keyfrms) {
    match::fuse matcher;

    {
        // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        auto cur_landmarks = cur_keyfrm->get_landmarks();
        for (const auto& fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            matcher.replace_duplication(fuse_tgt_keyfrm, cur_landmarks);
        }
    }

    // todo ivan. why do it again with the other way around? this is a heavy step, u got too much time?
    {
        // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        std::unordered_set<std::shared_ptr<data::landmark>> candidate_landmarks_to_fuse;
        candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * cur_keyfrm->num_keypts_);

        for (const auto& fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->get_landmarks();

            for (const auto& lm : fuse_tgt_landmarks) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }

                if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm))) {
                    continue;
                }
                candidate_landmarks_to_fuse.insert(lm);
            }
        }

        matcher.replace_duplication(cur_keyfrm, candidate_landmarks_to_fuse);
    }
}

void mapping_module::request_reset() {
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        reset_is_requested_ = true;
    }

    // BLOCK until reset
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx_reset_);
            if (!reset_is_requested_) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

bool mapping_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void mapping_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset mapping module");
    keyfrms_queue_.clear();
    local_map_cleaner_->reset();
    reset_is_requested_ = false;
}

void mapping_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
    std::lock_guard<std::mutex> lock2(mtx_keyfrm_queue_);
    abort_local_BA_ = true;
}

bool mapping_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

bool mapping_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_ && !force_to_run_;
}

void mapping_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause mapping module");
    is_paused_ = true;
}

bool mapping_module::set_force_to_run(const bool force_to_run) {
    std::lock_guard<std::mutex> lock(mtx_pause_);

    if (force_to_run && is_paused_) {
        return false;
    }

    force_to_run_ = force_to_run;
    return true;
}

void mapping_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    // clear the queue
    keyfrms_queue_.clear();

    spdlog::info("resume mapping module");
}

void mapping_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool mapping_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool mapping_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void mapping_module::terminate() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);
    is_paused_ = true;
    is_terminated_ = true;
}

template void mapping_module::fuse_landmark_duplication(const std::shared_ptr<data::keyframe> &cur_keyfrm,
                                               const vector<std::shared_ptr<data::keyframe>>& fuse_tgt_keyfrms);

} // namespace openvslam
