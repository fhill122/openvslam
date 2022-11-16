/*
 * Created by Ivan B on 2022/9/21.
 */

#ifndef SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_
#define SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_

#include <iostream>
#include "base.h"
#include <spdlog/spdlog.h>

namespace openvslam {
namespace camera{

struct CamOverlap{
    const int ind1, ind2;
    // black if only seen on cam1, gray if seen on cam2 as well
    cv::Mat mask;

    CamOverlap(const int ind_1, const int ind_2)
        : ind1(ind_1), ind2(ind_2) {}
};

struct CameraRig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //! raw pointer as to comply with the stupid openvslam config::camera_
    std::vector<std::unique_ptr<base>> cameras;
    //! T_b_c transform from camera to body
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses_inv;
    //! T_c_b saved to skip computation
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;

    std::vector<std::vector<CamOverlap>> overlaps;

    inline bool isMono(){
        CV_Assert(!cameras.empty());
        return cameras.size()==1 && cameras[0]->setup_type_==setup_type_t::Monocular;
    }

    void genOverlapMask(){
        CV_Assert(overlaps.size() == cameras.size());
        for (int i = 0; i < overlaps.size(); ++i) {
            for (int j=0; j<overlaps[i].size(); ++j) {
                CamOverlap &overlap = overlaps[i][j];
                CV_Assert(i==overlap.ind1);
                CV_Assert(overlap.ind2 >= 0 && overlap.ind2 <cameras.size());
                Eigen::Matrix3d R_2_1 = poses[overlap.ind2].block<3,3>(0,0)*
                                        poses_inv[overlap.ind1].block<3,3>(0,0);
                overlap.mask = cv::Mat(cameras[i]->rows_, cameras[i]->cols_, CV_8UC1);
                for (int v=0; v<overlap.mask.rows; ++v){
                    for (int u=0; u<overlap.mask.cols; ++u){
                        auto undist_pt = cameras[i]->undistort_point(cv::Point2f(u,v));
                        Eigen::Vector3d bearing = cameras[overlap.ind1]->convert_point_to_bearing(undist_pt);
                        Eigen::Vector2d projection;  // unused
                        float x_right; // unsused
                        bool inbound = cameras[overlap.ind2]->reproject_to_image(R_2_1, Eigen::Vector3d(0,0,0),
                                                                                 bearing, projection, x_right);
                        if (inbound){
                            overlap.mask.at<unsigned char>(v,u) = 127;
                        } else {
                            overlap.mask.at<unsigned char>(v,u) = 0;
                        }

                    }
                }
            }
        }
    }
};

}} // namespace

#endif // SRC_OPENVSLAM_CAMERA_CAMERA_RIG_H_
