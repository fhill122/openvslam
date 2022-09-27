/*
 * Created by Ivan B on 2022/9/21.
 */

#ifndef SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_
#define SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_

#include "base.h"
namespace openvslam {
namespace camera{

struct CameraRig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //! raw pointer as to comply with the stupid openvslam config::camera_
    std::vector<std::unique_ptr<base>> cameras;
    //! T_b_c transform from camera to body
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses_inv;
    //! T_c_b saved to skip computation
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;

    inline bool isMono(){
        CV_Assert(!cameras.empty());
        return cameras.size()==1 && cameras[0]->setup_type_==setup_type_t::Monocular;
    }
};

}} // namespace

#endif // SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_
