#ifndef OPENVSLAM_MODULE_LOCAL_MAP_UPDATER_H
#define OPENVSLAM_MODULE_LOCAL_MAP_UPDATER_H

#include <memory>
#include "openvslam/data/map_database.h"

namespace openvslam {

namespace data {
class frame;
class keyframe;
class landmark;
} // namespace data

namespace module {

class local_map_updater {
public:
    static std::vector<std::shared_ptr<data::landmark>>
    GetLocalLandmarks(const data::frame& curr_frm, const data::map_database *map_db, unsigned int num_kf){
        std::set<std::shared_ptr<data::landmark>> local_lms;

        auto kfs = map_db->getKeyframes<std::vector>(curr_frm.ref_keyfrm_->id_-num_kf, curr_frm.ref_keyfrm_->id_+1);

        for (const auto& keyfrm : kfs) {
            const auto& lms = keyfrm->get_landmarks();

            for (const auto& lm : lms) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }

                // todo ivan. shared_ptr copy maybe heavy
                local_lms.insert(lm);
            }
        }

        return {local_lms.begin(), local_lms.end()};
    }
};

} // namespace module
} // namespace openvslam

#endif // OPENVSLAM_MODULE_LOCAL_MAP_UPDATER_H
