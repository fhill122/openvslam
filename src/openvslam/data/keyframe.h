#ifndef OPENVSLAM_DATA_KEYFRAME_H
#define OPENVSLAM_DATA_KEYFRAME_H

#include "openvslam/type.h"
#include "openvslam/camera/base.h"
#include "openvslam/camera/cube_space.h"
#include "openvslam/data/bow_vocabulary.h"

#include <set>
#include <mutex>
#include <atomic>
#include <memory>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <nlohmann/json_fwd.hpp>

#ifdef USE_DBOW2
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#else
#include <fbow/bow_vector.h>
#include <fbow/bow_feat_vector.h>
#endif

namespace openvslam {

namespace camera {
class base;
} // namespace camera

namespace data {

class frame;
class landmark;
class map_database;
class MultiKeyframe;

class keyframe : public std::enable_shared_from_this<keyframe> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * Constructor for building from a frame
     */
    keyframe(MultiKeyframe *parent, unsigned int sibling_ind, const frame& frm, map_database* map_db);

    virtual ~keyframe();

    // Factory method for create keyframe
    static std::shared_ptr<keyframe> make_keyframe(MultiKeyframe *parent, unsigned int sibling_ind,
                                                   const frame& frm, map_database* map_db);

    // operator overrides
    bool operator==(const keyframe& keyfrm) const { return id_ == keyfrm.id_; }
    bool operator!=(const keyframe& keyfrm) const { return !(*this == keyfrm); }
    bool operator<(const keyframe& keyfrm) const { return id_ < keyfrm.id_; }
    bool operator<=(const keyframe& keyfrm) const { return id_ <= keyfrm.id_; }
    bool operator>(const keyframe& keyfrm) const { return id_ > keyfrm.id_; }
    bool operator>=(const keyframe& keyfrm) const { return id_ >= keyfrm.id_; }


    //-----------------------------------------
    // camera pose

    /**
     * Set camera pose
     */
    void set_cam_pose(const Mat44_t& cam_pose_cw);

    /**
     * Set camera pose
     */
    void set_cam_pose(const g2o::SE3Quat& cam_pose_cw);

    /**
     * Get the camera pose
     */
    Mat44_t get_cam_pose() const;

    /**
     * Get the inverse of the camera pose
     */
    Mat44_t get_cam_pose_inv() const;

    /**
     * Get the camera center
     */
    Vec3_t get_cam_center() const;

    /**
     * Get the rotation of the camera pose
     */
    Mat33_t get_rotation() const;

    /**
     * Get the translation of the camera pose
     */
    Vec3_t get_translation() const;

    //-----------------------------------------
    // features and observations

    /**
     * Compute BoW representation
     */
    void compute_bow();

    /**
     * Add a landmark observed by myself at keypoint idx
     */
    void add_landmark(std::shared_ptr<landmark> lm, const unsigned int idx);

    /**
     * Erase a landmark observed by myself at keypoint idx
     */
    void erase_landmark_with_index(const unsigned int idx);

    /**
     * Replace the landmark
     */
    void replace_landmark(std::shared_ptr<landmark>& lm, const unsigned int idx);

    /**
     * Get all of the landmarks
     * (NOTE: including nullptr)
     */
    std::vector<std::shared_ptr<landmark>> get_landmarks() const;

    /**
     * Get the valid landmarks
     */
    std::set<std::shared_ptr<landmark>> get_valid_landmarks() const;

    /**
     * Get the number of tracked landmarks which have observers equal to or greater than the threshold
     */
    unsigned int get_num_tracked_landmarks(const unsigned int min_num_obs_thr) const;

    /**
     * Get the landmark associated keypoint idx
     */
    std::shared_ptr<landmark>& get_landmark(const unsigned int idx);

    /**
     * Get the keypoint indices in the cell which reference point is located
     */
    std::vector<unsigned int> get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin) const;

    /**
     * Triangulate the keypoint using the disparity
     */
    Vec3_t triangulate_stereo(const unsigned int idx) const;

    /**
     * Compute median of depths
     */
    float compute_median_depth(const bool abs = false) const;

    /**
     * Whether or not the camera setting is capable of obtaining depth information
     */
    bool depth_is_avaliable() const;

    //-----------------------------------------
    // for local map update

    //! identifier for local map update
    unsigned int local_map_update_identifier = 0;


    //-----------------------------------------
    // meta information

    //! keyframe ID
    unsigned int id_;  // todo ivan. rm
    // todo [ivan] store reference? or make multikeyframe enable_shared_from_this?
    MultiKeyframe* const parent_;
    const unsigned int sibling_idx_;  // access index in parent
    //! next keyframe ID
    static std::atomic<unsigned int> next_id_; // todo ivan. rm

    //! timestamp in seconds
    const double timestamp_; // todo ivan. rm

    //-----------------------------------------
    // camera parameters

    //! camera model
    camera::base* camera_;
    //! depth threshold
    const float depth_thr_;

    //-----------------------------------------
    // constant observations

    //! number of keypoints
    const unsigned int num_keypts_;

    //! keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> keypts_;
    //! undistorted keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> undist_keypts_;
    //! if using cube space
    std::vector<camera::CubeSpace::CubePoint> cube_keypts_;
    //! bearing vectors
    const eigen_alloc_vector<Vec3_t> bearings_;

    //! keypoint indices in each of the cells
    const std::vector<std::vector<std::vector<unsigned int>>> keypt_indices_in_cells_;

    //! disparities
    const std::vector<float> stereo_x_right_;
    //! depths
    const std::vector<float> depths_;

    //! descriptors
    const cv::Mat descriptors_;

    //! BoW features (DBoW2 or FBoW)
    // [ivan] still used for triangulation match
#ifdef USE_DBOW2
    DBoW2::BowVector bow_vec_;
    DBoW2::FeatureVector bow_feat_vec_;
#else
    fbow::BoWVector bow_vec_;
    fbow::BoWFeatVector bow_feat_vec_;
#endif

    //-----------------------------------------
    // ORB scale pyramid information

    //! number of scale levels
    const unsigned int num_scale_levels_;
    //! scale factor
    const float scale_factor_;
    //! log scale factor
    const float log_scale_factor_;
    //! list of scale factors
    const std::vector<float> scale_factors_;
    //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
    const std::vector<float> level_sigma_sq_;
    //! list of 1 / sigma^2 for optimization
    const std::vector<float> inv_level_sigma_sq_;

    //-----------------------------------------
    // databases

    //! map database
    map_database* map_db_;
    //! BoW vocabulary
    bow_vocabulary* bow_vocab_;

private:
    //-----------------------------------------
    // camera pose

    //! need mutex for access to poses
    mutable std::mutex mtx_pose_;
    //! camera pose from the world to the current
    Mat44_t cam_pose_cw_;
    //! camera pose from the current to the world
    Mat44_t cam_pose_wc_;
    //! camera center
    Vec3_t cam_center_;

    //-----------------------------------------
    // observations

    //! need mutex for access to observations
    mutable std::mutex mtx_observations_;
    //! observed landmarks
    std::vector<std::shared_ptr<landmark>> landmarks_;

    //-----------------------------------------
    // flags

    //! flag which indicates this keyframe is erasable or not
    std::atomic<bool> cannot_be_erased_{false};

    //! flag which indicates this keyframe will be erased
    std::atomic<bool> will_be_erased_{false};
};

} // namespace data
} // namespace openvslam

#endif // OPENVSLAM_DATA_KEYFRAME_H
