#ifndef OPENVSLAM_MODULE_LOCAL_MAP_CLEANER_H
#define OPENVSLAM_MODULE_LOCAL_MAP_CLEANER_H

#include <list>
#include <memory>

namespace openvslam {

namespace data {
class keyframe;
class landmark;
} // namespace data

namespace module {

class local_map_cleaner {
public:
    /**
     * Constructor
     */
    explicit local_map_cleaner(double redundant_obs_ratio_thr = 0.9);

    /**
     * Destructor
     */
    ~local_map_cleaner() = default;
    
    /**
     * Add fresh landmark to check their redundancy
     */
    void add_fresh_landmark(std::shared_ptr<data::landmark>& lm) {
        fresh_landmarks_.push_back(lm);
    }

    /**
     * Reset the buffer
     */
    void reset();

    /**
     * Remove redundant landmarks
     */
    unsigned int remove_redundant_landmarks(const unsigned int cur_keyfrm_id, const bool depth_is_avaliable);
    
private:
    //!
    double redundant_obs_ratio_thr_;

    //! fresh landmarks to check their redundancy
    std::list<std::shared_ptr<data::landmark>> fresh_landmarks_;
};

} // namespace module
} // namespace openvslam

#endif // OPENVSLAM_MODULE_LOCAL_MAP_CLEANER_H