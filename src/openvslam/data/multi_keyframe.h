/*
 * Created by Ivan B on 2022/9/23.
 */

#ifndef SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_
#define SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_

#include "keyframe.h"
#include "multi_frame.h"
#include "openvslam/camera/camera_rig.h"

namespace openvslam {
namespace data{

struct MultiKeyframe : std::enable_shared_from_this<MultiKeyframe>{
    // todo [ivan] remove tailing _
    //! keyframe ID
    unsigned int id_;
    //! next keyframe ID
    inline static std::atomic<unsigned int> next_id_{0};

    // todo ivan. should be unique_ptr
    std::vector<std::shared_ptr<keyframe>> frames;
    camera::CameraRig* rig;

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

private:
    // MultiKeyframe() = default;
    // todo [ivan] should take frames?
    MultiKeyframe(camera::CameraRig* rig): id_(next_id_.fetch_add(std::memory_order_relaxed)), rig(rig){
        frames.reserve(rig->cameras.size());
    }

    MultiKeyframe(const MultiFrame &m_frame, map_database* map_db) :
          id_(next_id_.fetch_add(std::memory_order_relaxed)), rig(m_frame.rig){
        for (int i=0; i<m_frame.frames.size(); ++i) {
            frames.emplace_back(keyframe::make_keyframe(this, i, *m_frame.frames[i], map_db));
        }
    }


public:
    [[nodiscard]] static std::shared_ptr<MultiKeyframe> Create(const MultiFrame &m_frame, map_database* map_db){
        return std::shared_ptr<MultiKeyframe>(new MultiKeyframe(m_frame, map_db));
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

    inline std::shared_ptr<keyframe>& operator[](int i){return frames.at(i);}

    inline std::shared_ptr<keyframe>& at(int i){return frames.at(i);}

    // inline std

    inline int size() const { return frames.size();}

    /* poses api */

    inline void setFramesPoses(){
        AssertLog(cam_pose_cw_is_valid_, "");
        for (int i=0; i<frames.size(); ++i) {
            frames[i]->set_cam_pose(rig->poses[i]*cam_pose_cw_);
        }
    }

    inline void setCamPose(const Mat44_t& cam_pose_cw){
        cam_pose_cw_is_valid_ = true;
        cam_pose_cw_ = cam_pose_cw;
        rot_cw_ = cam_pose_cw_.block<3, 3>(0, 0);
        rot_wc_ = rot_cw_.transpose();
        trans_cw_ = cam_pose_cw_.block<3, 1>(0, 3);
        cam_center_ = -rot_cw_.transpose() * trans_cw_;
        setFramesPoses();
    }

    inline void setCamPose(const g2o::SE3Quat& cam_pose_cw) {
        setCamPose(util::converter::to_eigen_mat(cam_pose_cw));
    }

    inline Mat44_t getCamPose() const {
        return cam_pose_cw_;
    }

    inline Mat44_t getCamPoseInv() const {
        Mat44_t cam_pose_wc = Mat44_t::Identity();
        cam_pose_wc.block<3, 3>(0, 0) = rot_wc_;
        cam_pose_wc.block<3, 1>(0, 3) = cam_center_;
        return cam_pose_wc;
    }

    inline Vec3_t getCamCenter() const {
        return cam_center_;
    }

    inline Mat33_t getRotationInv() const {
        return rot_wc_;
    }

    /* api that calls each frame */
    unsigned int getNumTrackedLm(unsigned int min_obs_thr) const{
        unsigned int total = 0;
        for (const auto& frm : frames){
            total += frm->get_num_tracked_landmarks(min_obs_thr);
        }
    }
};

}} // namespace openvslam

#endif // SRC_OPENVSLAM_DATA_MULTI_KEYFRAME_H_
