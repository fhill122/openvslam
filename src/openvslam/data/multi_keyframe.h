/*
 * Created by Ivan B on 2022/9/23.
 */

#ifndef SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_
#define SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_

#include "keyframe.h"
#include "openvslam/camera/camera_rig.h"

namespace openvslam {
namespace data{

struct MultiKeyframe {
    std::vector<std::shared_ptr<keyframe>> frames;
    camera::CameraRig* rig;

    //! keyframe ID
    unsigned int id_;
    //! next keyframe ID
    static std::atomic<unsigned int> next_id_;

    //! timestamp in seconds
    double timestamp_;

    // Camera rig pose

    bool cam_pose_cw_is_valid_ = false;
    //! camera pose: world -> camera
    Mat44_t cam_pose_cw_;
    //! rotation: world -> camera
    Mat33_t rot_cw_;
    //! translation: world -> camera
    Vec3_t trans_cw_;
    //! rotation: camera -> world
    Mat33_t rot_wc_;
    //! translation: camera -> world
    Vec3_t cam_center_;



    //////////////////////////////////////////////////////

    MultiKeyframe() = default;
    MultiKeyframe(camera::CameraRig* rig): rig(rig){
        frames.reserve(rig->cameras.size());
    }

    // todo ivan. reconsider this
    Vec3_t spacialTriangulate(){

    }

    // operator overrides
    bool operator==(const MultiKeyframe& keyfrm) const { return id_ == keyfrm.id_; }
    bool operator!=(const MultiKeyframe& keyfrm) const { return !(*this == keyfrm); }
    bool operator<(const MultiKeyframe& keyfrm) const { return id_ < keyfrm.id_; }
    bool operator<=(const MultiKeyframe& keyfrm) const { return id_ <= keyfrm.id_; }
    bool operator>(const MultiKeyframe& keyfrm) const { return id_ > keyfrm.id_; }
    bool operator>=(const MultiKeyframe& keyfrm) const { return id_ >= keyfrm.id_; }

    /* poses api */

    inline void setFramesPoses(){
        AssertLog(cam_pose_cw_is_valid_, "");
        for (int i=0; i<frames.size(); ++i) {
            frames[i]->set_cam_pose(rig->poses[i]*cam_pose_cw_);
        }
    }

    inline void set_cam_pose(const Mat44_t& cam_pose_cw){
        cam_pose_cw_is_valid_ = true;
        cam_pose_cw_ = cam_pose_cw;
        rot_cw_ = cam_pose_cw_.block<3, 3>(0, 0);
        rot_wc_ = rot_cw_.transpose();
        trans_cw_ = cam_pose_cw_.block<3, 1>(0, 3);
        cam_center_ = -rot_cw_.transpose() * trans_cw_;
        setFramesPoses();
    }

    inline void set_cam_pose(const g2o::SE3Quat& cam_pose_cw) {
        set_cam_pose(util::converter::to_eigen_mat(cam_pose_cw));
    }

    inline Mat44_t get_cam_pose() const {
        return cam_pose_cw_;
    }

    inline Mat44_t get_cam_pose_inv() const {
        Mat44_t cam_pose_wc = Mat44_t::Identity();
        cam_pose_wc.block<3, 3>(0, 0) = rot_wc_;
        cam_pose_wc.block<3, 1>(0, 3) = cam_center_;
        return cam_pose_wc;
    }

    inline Vec3_t get_cam_center() const {
        return cam_center_;
    }

    inline Mat33_t get_rotation_inv() const {
        return rot_wc_;
    }
};

}} // namespace openvslam

#endif // SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_
