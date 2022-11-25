/*
 * Created by Ivan B on 2022/11/8.
 */

#include <spdlog/spdlog.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include "root.h"
#include "openvslam/config.h"
#include "openvslam/module/initializer.h"
#include "openvslam/data/map_database.h"
#include "openvslam/util/yaml.h"
#include "openvslam/data/multi_frame.h"
#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/feature/orb_extractor.h"
#include "ivtb/log.h"
#include "ivtb/periodic_runner.h"

using namespace std;
namespace ov = openvslam;

const vector<string> kImgs = {
    "/Users/ivan/data/kitti/odom/sequences/00/image_0/000000.png",
    "/Users/ivan/data/kitti/odom/sequences/00/image_1/000000.png"};
const string kConfig = string(kRootDir) + "/example/kitti/KITTI_multi_00-02_cubespace.yaml";
const string kVocab = string(kRootDir) + "/data/orb_vocab.fbow";

void PubCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, ros::Publisher &pub){
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = ros::Time::now();
    pub.publish(msg);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "triang_init");
    ros::NodeHandle nh("~");
    ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("map", 1, true);
    spdlog::set_level(spdlog::level::debug);

    auto config = make_shared<ov::config>(string(kConfig));
    auto map_db = make_unique<ov::data::map_database>();
    AssertLog(config->cam_rig_->cameras.size()==kImgs.size(), "");

    ov::module::initializer initializer{false, map_db.get(),
                                        ov::util::yaml_optional_ref(config->yaml_node_, "Initializer")};
    ov::data::MultiFrame frame{config->cam_rig_.get()};
    ov::feature::orb_extractor extractor{2000, 1.2, 8, 20, 7};
    fbow::Vocabulary bow_vocab;
    bow_vocab.readFromFile(kVocab);

    // create frame
    for (int i = 0; i < kImgs.size(); ++i) {
        cv::Mat img = cv::imread(kImgs[i]);
        if (img.channels()==3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        frame.frames.push_back(shared_ptr<ov::data::frame>(new ov::data::frame(
            img, 0, &extractor, &bow_vocab, config->cam_rig_->cameras[i].get(), 0, cv::Mat()
        )));
    }

    // initialize
    bool ok = initializer.initialize(frame);
    spdlog::info("init result: {}", ok);

    // visualize
    auto kf = map_db->getKeyframe(0);
    AssertLog(kf, "");
    // map
    pcl::PointCloud<pcl::PointXYZ> points;
    auto lms = map_db->get_all_landmarks();
    points.reserve(lms.size());
    for(const auto &lm : lms){
        ov::Vec3_t pos = lm->get_pos_in_world();
        points.push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
        RUN_N_TIMES(20,spdlog::debug("add a point of ({},{},{})", pos.x(), pos.y(), pos.z()));
    }
    PubCloud(points, map_pub);
    spdlog::info("pub a map of {} points", points.size());
    // image
    for (int i = 0; i < frame.size(); ++i) {
        cv::Mat draw;
        cv::drawKeypoints(frame[i]->img_, frame[i]->keypts_, draw);
        cv::imshow("frame" + to_string(i), draw);
    }
    // show match
    for (int i=0; i<kf->size(); ++i) {
        for (int j = 0; j < frame.rig->overlaps[i].size(); ++j) {
            const auto& overlap = frame.rig->overlaps[i][j];
            vector<cv::DMatch> match;
            match.reserve(frame[overlap.ind1]->num_keypts_);
            for(int k1 =0; k1 <frame[overlap.ind1]->num_keypts_; ++k1){
                auto &lm = frame[overlap.ind1]->landmarks_[k1];
                if(lm){
                    int k2 = lm->get_index_in_keyframe(kf->at(overlap.ind2));
                    if (k2>=0) match.push_back({k1,k2,0});
                }
            }
            cv::Mat match_img;
            cv::drawMatches(frame[overlap.ind1]->img_, frame[overlap.ind1]->keypts_,
                            frame[overlap.ind2]->img_, frame[overlap.ind2]->keypts_,
                            match, match_img);
            cv::imshow("match_" + to_string(overlap.ind1) + "_" + to_string(overlap.ind2), match_img);
        }
    }

    cv::waitKey();
    // this_thread::sleep_for(10s);
    return 0;
}
