/*
 * note :
 * openmp has good effect
 * for every fast call, adding a clone increase the total time by 0.2-0.4 ms
 * tree distribute only 80% points, then get best 20% of the other points
 *
 * Created by Ivan B on 2021/12/17.
 */

#include <opencv2/opencv.hpp>

#include "../orb_extractor.h"
#include "common/log.h"
#include "common/stopwatch.h"
#include "root.h"

using namespace openvslam;
using namespace openvslam::feature;
using namespace std;
using namespace ivtb;


int main(int argc, char** argv){
    // cv::Mat test(640, 480, CV_8UC1, cv::Scalar{0});
    // printf("value at 1,1 = %d\n", test.at<uchar>(1,1));
    // test.rowRange(0, 10).colRange(0, 10).at<uchar>(1,1) = 1;
    // printf("value at 1,1 = %d\n", test.at<uchar>(1,1));
    // cv::Mat temp = test.rowRange(0, 10).colRange(0, 10).clone();
    // temp.at<uchar>(1,1) = 2;
    // printf("value at 1,1 = %d\n", test.at<uchar>(1,1));
    // printf("value at 1,1 = %d\n", temp.at<uchar>(1,1));


    Log::D(__func__, "start");
    auto src_img = cv::imread(string(kRootDir)+"/data/tum1.png");
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);
    // cv::resize(src_img, src_img, {}, 0.5, 0.5);

    orb_params params;
    params.num_levels_ = 1;
    // two threshold just to speed up?
    params.ini_fast_thr_ = 20;
    params.min_fast_thr = 7;
    orb_extractor extractor{500, params};

    vector<cv::KeyPoint> keypts;
    cv::Mat descriptors;
    Stopwatch stopwatch;
    extractor.extract(src_img, cv::Mat(), keypts, descriptors);
    Log::I(__func__, "extraction time: %.1f ms, %lu points", stopwatch.passedMs(), keypts.size());

    cv::Mat img_show;
    cv::drawKeypoints(src_img,keypts, img_show);
    cv::imshow("orb_bench", img_show);
    cv::waitKey();

    return 0;
}
