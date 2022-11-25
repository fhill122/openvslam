#ifndef OPENVSLAM_MODULE_FRAME_TRACKER_H
#define OPENVSLAM_MODULE_FRAME_TRACKER_H

#include "openvslam/type.h"
#include "openvslam/optimize/pose_optimizer.h"

#include <memory>

namespace openvslam {

namespace camera {
class base;
} // namespace camera

namespace data {
struct MultiFrame;
struct MultiKeyframe;
} // namespace data

namespace module {

class frame_tracker {
public:
    explicit frame_tracker(const unsigned int num_matches_thr = 20);

    bool motion_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                            const Mat44_t& velocity) const;

    bool bow_match_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                               const std::shared_ptr<data::MultiKeyframe>& ref_keyfrm) const;

    bool robust_match_based_track(data::MultiFrame& curr_frm, const data::MultiFrame& last_frm,
                                  const std::shared_ptr<data::MultiKeyframe>& ref_keyfrm) const;

private:
    // unsigned int discard_outliers(data::MultiFrame& curr_frm) const;

    const unsigned int num_matches_thr_;

    const optimize::pose_optimizer pose_optimizer_;
};

} // namespace module
} // namespace openvslam

#endif // OPENVSLAM_MODULE_FRAME_TRACKER_H