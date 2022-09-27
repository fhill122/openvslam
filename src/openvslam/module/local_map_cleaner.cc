#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/module/local_map_cleaner.h"

namespace openvslam {
namespace module {

local_map_cleaner::local_map_cleaner(double redundant_obs_ratio_thr)
    : redundant_obs_ratio_thr_(redundant_obs_ratio_thr) {}

void local_map_cleaner::reset() {
    fresh_landmarks_.clear();
}

unsigned int local_map_cleaner::remove_redundant_landmarks(const unsigned int cur_keyfrm_id, const bool depth_is_avaliable) {
    constexpr float observed_ratio_thr = 0.3;
    constexpr unsigned int num_reliable_keyfrms = 2;
    const unsigned int num_obs_thr = depth_is_avaliable ? 3 : 2;

    // states of observed landmarks
    enum class lm_state_t { Valid,
                            Invalid,
                            NotClear };

    unsigned int num_removed = 0;
    auto iter = fresh_landmarks_.begin();
    while (iter != fresh_landmarks_.end()) {
        const auto& lm = *iter;

        // decide the state of lms the buffer
        auto lm_state = lm_state_t::NotClear;
        if (lm->will_be_erased()) {
            // in case `lm` will be erased
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }
        else if (lm->get_observed_ratio() < observed_ratio_thr) {
            // if `lm` is not reliable
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + lm->first_keyfrm_id_ <= cur_keyfrm_id
                 && lm->num_observations() <= num_obs_thr) {
            // if the number of the observers of `lm` is small after some keyframes were inserted
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + 1U + lm->first_keyfrm_id_ <= cur_keyfrm_id) {
            // if the number of the observers of `lm` is sufficient after some keyframes were inserted
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }

        // select to remove `lm` according to the state
        if (lm_state == lm_state_t::Valid) {
            iter = fresh_landmarks_.erase(iter);
        }
        else if (lm_state == lm_state_t::Invalid) {
            ++num_removed;
            lm->prepare_for_erasing();
            iter = fresh_landmarks_.erase(iter);
        }
        else {
            // hold decision because the state is NotClear
            iter++;
        }
    }

    return num_removed;
}



} // namespace module
} // namespace openvslam
