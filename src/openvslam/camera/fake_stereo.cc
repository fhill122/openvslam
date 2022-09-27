/*
 * Created by Ivan B on 2022/9/1.
 */

#include <opencv2/opencv.hpp>
#include "fake_stereo.h"
#include "openvslam/solve/triangulator.h"

using namespace std;

namespace openvslam {

void FakeStereo::FindStereo(const cv::Mat& img1, const cv::Mat& img2,
                            const std::vector<cv::Point2f>& keypts_1,
                            const eigen_alloc_vector<Vec3_t>& bearings_1,
                            const Mat33_t& rot_21, const Vec3_t& trans_21,
                            const camera::base* cam2, double virtual_focal,
                            double& base_line, std::vector<double>& ur) {
    // todo ivan. using cubespace, we actually allow negative ur, but that's not the case where other part of the openvslam
    vector<cv::Point3f> pts3d;
    FindStereo(img1, img2, keypts_1, bearings_1, rot_21, trans_21, cam2, pts3d);

    ur.clear();
    ur.reserve(pts3d.size());
    base_line = trans_21.norm();
    for(size_t i = 0; i < pts3d.size(); i++) {
        if (pts3d[i].z==0){
            ur.push_back(DBL_MAX);
            continue;
        }
        ur.push_back(keypts_1[i].x - virtual_focal*base_line/pts3d[i].z);
    }
}

void FakeStereo::FindStereo(const cv::Mat& img1, const cv::Mat& img2,
                            const vector<cv::Point2f>& keypts_1, const eigen_alloc_vector<Vec3_t>& bearings_1,
                            const Mat33_t& rot_21, const Vec3_t& trans_21, const camera::base* cam2,
                            vector<cv::Point3f>& pts3d) {
    constexpr double kReverseErrTh = 0.5;
    constexpr double kReprojErrTh = 2.0;

    auto distance_func = [](const cv::Point2f &pt1, const cv::Point2f &pt2)->bool
    {return hypot(pt1.x - pt2.x, pt1.y - pt2.y);};

    // todo ivan. timing it

    vector<cv::Point2f> keypts_2;
    vector<cv::Point2f> keypts_1_reversed;
    vector<uchar> status;
    vector<uchar> status_reversed;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, keypts_1, keypts_2, status, err, cv::Size(21, 21), 3);

    // todo ivan. build pyramid and reuse it
    // reverse check cur right ---- cur left
    cv::calcOpticalFlowPyrLK(img2, img1, keypts_2, keypts_1_reversed, status_reversed, err, cv::Size(21, 21), 3);
    for(size_t i = 0; i < status.size(); i++){
        if(status[i] && status_reversed[i] && distance_func(keypts_1[i], keypts_1_reversed[i]) <= kReverseErrTh){
            status[i] = 1;
            CV_Assert(keypts_1[i].x > 0 && keypts_1[i].x < img1.cols-1 &&
                      keypts_1[i].y > 0 && keypts_1[i].y < img1.rows-1);
        }
        else
            status[i] = 0;
    }

    pts3d.clear();
    pts3d.reserve(status.size());
    for(size_t i = 0; i < status.size(); i++) {
        if (status[i]==0){
            pts3d.emplace_back(0,0,0);
            continue;
        }
        Vec3_t bearing_2 = cam2->convert_point_to_bearing(keypts_2[i]);
        Vec3_t pt3d = solve::triangulator::triangulate(bearings_1[i], bearing_2, rot_21, trans_21);
        cv::Point2f reproj = cam2->convert_bearing_to_point(pt3d);
        if (distance_func(keypts_2[i], reproj)>kReprojErrTh) {
            pts3d.emplace_back(0,0,0);
            continue ;
        }
        pts3d.emplace_back(pt3d.x(),pt3d.y(),pt3d.z());
    }
}

}
