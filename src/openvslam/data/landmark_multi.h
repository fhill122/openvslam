#ifndef OPENVSLAM_DATA_LANDMARK_H
#define OPENVSLAM_DATA_LANDMARK_H

#include "openvslam/type.h"

#include <map>
#include <mutex>
#include <atomic>
#include <memory>

#include <opencv2/core/core.hpp>
#include <nlohmann/json_fwd.hpp>

namespace openvslam {
namespace data {

class frame;

class MultiKeyframe;

class map_database;

struct Observation{
    std::weak_ptr<MultiKeyframe> keyfrm;
    // frame index, keypoint index. expect small size
    std::map<unsigned int, unsigned int> frm_kp;
};

struct ObservationBook{
    std::unordered_map<unsigned int, Observation> observations;
    unsigned int num_observations = 0;

    void add(const std::shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx, unsigned int kp_idx);

    void erase(const std::shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx);
};

class landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // map instead of unordered_map as erase_observation needs another
    // using observations_t = std::map<std::weak_ptr<keyframe>, unsigned int,
    //                                 std::owner_less<std::weak_ptr<keyframe>>>;

    //! constructor
    landmark(const Vec3_t& pos_w, const Observation& ref_obs, map_database* map_db);

    //! set world coordinates of this landmark
    void set_pos_in_world(const Vec3_t& pos_w);
    //! get world coordinates of this landmark
    Vec3_t get_pos_in_world() const;

    //! get mean normalized vector of keyframe->lm vectors, for keyframes such that observe the 3D point.
    Vec3_t get_obs_mean_normal() const;

    //! add observation
    void add_observation(const std::shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx, unsigned int kp_idx);
    //! erase observation
    void erase_observation(const std::shared_ptr<MultiKeyframe>& keyfrm, unsigned int frm_idx);

    //! get observations (keyframe and keypoint idx)
    ObservationBook get_observations() const;
    //! get number of observations
    unsigned int num_observations() const;
    //! whether this landmark is observed from more than zero keyframes
    bool has_observation() const;

    //! whether this landmark is observed in the specified keyframe
    bool is_observed_in_keyframe(const std::shared_ptr<keyframe>& keyfrm) const;

    //! check the distance between landmark and camera is in ORB scale variance
    inline bool is_inside_in_orb_scale(const float cam_to_lm_dist) const {
        const float max_dist = this->get_max_valid_distance();
        const float min_dist = this->get_min_valid_distance();
        return (min_dist <= cam_to_lm_dist && cam_to_lm_dist <= max_dist);
    }

    //! get representative descriptor
    cv::Mat get_descriptor() const;

    //! compute representative descriptor
    void compute_descriptor();

    //! update observation mean normal and ORB scale variance
    void update_normal_and_depth();

    //! get max valid distance between landmark and camera
    float get_min_valid_distance() const;
    //! get min valid distance between landmark and camera
    float get_max_valid_distance() const;

    //! predict scale level assuming this landmark is observed in the specified frame
    unsigned int predict_scale_level(const float cam_to_lm_dist, const frame* frm) const;
    //! predict scale level assuming this landmark is observed in the specified keyframe
    unsigned int predict_scale_level(const float cam_to_lm_dist, const std::shared_ptr<keyframe>& keyfrm) const;

    //! erase this landmark from database
    void prepare_for_erasing();
    //! whether this landmark will be erased shortly or not
    bool will_be_erased();

    //! replace this with specified landmark
    void replace(std::shared_ptr<landmark> lm);
    //! get replace landmark
    std::shared_ptr<landmark> get_replaced() const;

    // note ivan. these are called upon each frame tracking. these are used for observed ratio, so count
    // based on multiframe observation
    void increase_num_observable(unsigned int num_observable = 1);
    void increase_num_observed(unsigned int num_observed = 1);
    float get_observed_ratio() const;


public:
    unsigned int id_;
    static std::atomic<unsigned int> next_id_;
    unsigned int first_keyfrm_id_ = 0;

    // Variables for frame tracking.
    Vec2_t reproj_in_tracking_;
    float x_right_in_tracking_;
    // todo ivan. very bad naming.
    bool is_observable_in_tracking_;
    int scale_level_in_tracking_;
    unsigned int identifier_in_local_map_update_ = 0;
    unsigned int identifier_in_local_lm_search_ = 0;

private:
    //! world coordinates of this landmark
    Vec3_t pos_w_;

    //! observations (keyframe and keypoint index)
    ObservationBook observations_;

    //! Normalized average vector (unit vector) of keyframe->lm, for keyframes such that observe the 3D point.
    Vec3_t mean_normal_ = Vec3_t::Zero();

    //! representative descriptor
    cv::Mat descriptor_;

    //! reference keyframe
    // std::weak_ptr<keyframe> ref_keyfrm_;
    Observation ref_obs_;

    // track counter
    unsigned int num_observable_ = 1;
    unsigned int num_observed_ = 1;

    //! this landmark will be erased shortly or not
    bool will_be_erased_ = false;

    //! replace this landmark with below
    std::shared_ptr<landmark> replaced_ = nullptr;

    // ORB scale variances
    //! max valid distance between landmark and camera
    float min_valid_dist_ = 0;
    //! min valid distance between landmark and camera
    float max_valid_dist_ = 0;

    //! map database
    map_database* map_db_;

    mutable std::mutex mtx_position_;
    mutable std::mutex mtx_observations_;
};

} // namespace data
} // namespace openvslam

#endif // OPENVSLAM_DATA_LANDMARK_H
