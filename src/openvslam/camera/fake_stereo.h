/*
 * Created by Ivan B on 2022/9/1.
 */

#ifndef SRC_OPENVSLAM_CAMERA_FACE_STEREO_H_
#define SRC_OPENVSLAM_CAMERA_FACE_STEREO_H_

#include <opencv2/core.hpp>
#include "base.h"

namespace openvslam{

class FakeStereo {


    /**
     * use klt to find right points, convert it to depth.
     * note we could have 3d locations for keypoints, making it predictable on the right
     * so many improvements to go. should take pyramid instead of image. flow back check?
     * what check should we do instead of epipolar line
     * @param img1
     * @param img2
     * @param keypts_1
     * @param bearings_1
     * @param rot_21
     * @param trans_21
     * @param cam2
     * @param virtual_focal
     * @param base_line output
     * @param ur output
     */
    void FindStereo(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f>& keypts_1,
                    const eigen_alloc_vector<Vec3_t> &bearings_1,
                    const Mat33_t& rot_21, const Vec3_t& trans_21,
                    const camera::base* cam2, double virtual_focal,
                    double &base_line, std::vector<double> &ur);
    /**
     *
     * @param img1
     * @param img2
     * @param keypts_1
     * @param bearings_1
     * @param rot_21
     * @param trans_21
     * @param cam2
     * @param pts3d output
     */
    void FindStereo(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f>& keypts_1,
                    const eigen_alloc_vector<Vec3_t> &bearings_1,
                    const Mat33_t& rot_21, const Vec3_t& trans_21,
                    const camera::base* cam2,
                    std::vector<cv::Point3f> &pts3d);
};

}

#endif // SRC_OPENVSLAM_CAMERA_FACE_STEREO_H_
