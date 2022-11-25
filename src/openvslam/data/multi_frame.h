/*
 * Created by Ivan B on 2022/9/23.
 */

#ifndef SRC_OPENVSLAM_DATA_MULTI_FRAME_H_
#define SRC_OPENVSLAM_DATA_MULTI_FRAME_H_

#include "frame.h"
#include "openvslam/camera/camera_rig.h"


namespace openvslam {
namespace data{


struct MultiKeyframe;

struct MultiFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! current frame ID
    unsigned int id_;
    //! next frame ID
    inline static std::atomic<unsigned int> next_id_{0};

    std::vector<std::shared_ptr<frame>> frames;
    camera::CameraRig* rig;

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

    //! reference keyframe for tracking
    std::shared_ptr<MultiKeyframe> ref_keyfrm_ = nullptr;

    //////////////////////////////////////////////////////

    MultiFrame() = delete;

    explicit MultiFrame(unsigned int id) : id_(id){};

    explicit MultiFrame(camera::CameraRig* rig): id_(next_id_.fetch_add(1, std::memory_order_relaxed)), rig(rig){
        frames.reserve(rig->cameras.size());
    }

    bool operator==(const frame& frm) const { return this->id_ == frm.id_; }
    bool operator!=(const frame& frm) const { return !(*this == frm); }

    inline std::shared_ptr<frame>& operator[](int i){return frames.at(i);}

    inline std::shared_ptr<frame>& at(int i){return frames.at(i);}

    inline int size(){ return frames.size();}

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

    /* api that calls each frame */
    unsigned int getNumKeypts() const{
        unsigned int n =0;
        for (const auto &frm : frames) {
            n += frm->num_keypts_;
        }
        return n;
    }

    std::vector<std::vector<cv::KeyPoint>> getKeypts() const{
        std::vector<std::vector<cv::KeyPoint>> out(frames.size());
        for (int i=0; i<frames.size(); ++i){
            out[i] = frames[i]->keypts_;
        }
        return out;
    }
};

}} // namespace

#endif // SRC_OPENVSLAM_DATA_MULTI_FRAME_H_
