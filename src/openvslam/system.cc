#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/tracking_module.h"
#include "openvslam/mapping_module.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/map_database.h"
#include "openvslam/data/bow_vocabulary.h"
#include "openvslam/io/trajectory_io.h"
#include "openvslam/publish/map_publisher.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/util/yaml.h"

#include <thread>

#include <spdlog/spdlog.h>

namespace openvslam {

system::system(const std::shared_ptr<config>& cfg, const std::string& vocab_file_path)
    : cfg_(cfg), cam_rig_(cfg->cam_rig_.get()) {
    spdlog::debug("CONSTRUCT: system");

    std::ostringstream message_stream;

    message_stream << std::endl;
    message_stream << R"(  ___               __   _____ _      _   __  __ )" << std::endl;
    message_stream << R"( / _ \ _ __  ___ _ _\ \ / / __| |    /_\ |  \/  |)" << std::endl;
    message_stream << R"(| (_) | '_ \/ -_) ' \\ V /\__ \ |__ / _ \| |\/| |)" << std::endl;
    message_stream << R"( \___/| .__/\___|_||_|\_/ |___/____/_/ \_\_|  |_|)" << std::endl;
    message_stream << R"(      |_|                                        )" << std::endl;
    message_stream << std::endl;
    message_stream << "Copyright (C) 2019," << std::endl;
    message_stream << "National Institute of Advanced Industrial Science and Technology (AIST)" << std::endl;
    message_stream << "All rights reserved." << std::endl;
    message_stream << std::endl;
    message_stream << "This is free software," << std::endl;
    message_stream << "and you are welcome to redistribute it under certain conditions." << std::endl;
    message_stream << "See the LICENSE file." << std::endl;
    message_stream << std::endl;

    // show configuration
    message_stream << *cfg_ << std::endl;

    spdlog::info(message_stream.str());

    // load ORB vocabulary
    spdlog::info("loading ORB vocabulary: {}", vocab_file_path);
#ifdef USE_DBOW2
    bow_vocab_ = new data::bow_vocabulary();
    try {
        bow_vocab_->loadFromBinaryFile(vocab_file_path);
    }
    catch (const std::exception&) {
        spdlog::critical("wrong path to vocabulary");
        delete bow_vocab_;
        bow_vocab_ = nullptr;
        exit(EXIT_FAILURE);
    }
#else
    bow_vocab_ = new fbow::Vocabulary();
    bow_vocab_->readFromFile(vocab_file_path);
    if (!bow_vocab_->isValid()) {
        spdlog::critical("wrong path to vocabulary");
        delete bow_vocab_;
        bow_vocab_ = nullptr;
        exit(EXIT_FAILURE);
    }
#endif

    // database
    map_db_ = new data::map_database();
    auto bow_database_yaml_node = util::yaml_optional_ref(cfg->yaml_node_, "BowDatabase");
    int reject_by_graph_distance = bow_database_yaml_node["reject_by_graph_distance"].as<bool>(false);
    int loop_min_distance_on_graph = bow_database_yaml_node["loop_min_distance_on_graph"].as<int>(30);

    // frame and map publisher
    frame_publisher_ = std::shared_ptr<publish::frame_publisher>(new publish::frame_publisher(cfg_, map_db_));
    map_publisher_ = std::shared_ptr<publish::map_publisher>(new publish::map_publisher(cfg_, map_db_));

    // tracking module
    tracker_ = new tracking_module(cfg_, this, map_db_, bow_vocab_);
    // mapping module
    mapper_ = new mapping_module(cfg_->yaml_node_["Mapping"], map_db_);

    // connect modules each other
    tracker_->set_mapping_module(mapper_);
    mapper_->set_tracking_module(tracker_);
}

system::~system() {
    mapping_thread_.reset(nullptr);
    delete mapper_;
    mapper_ = nullptr;

    delete tracker_;
    tracker_ = nullptr;

    delete map_db_;
    map_db_ = nullptr;
    delete bow_vocab_;
    bow_vocab_ = nullptr;

    spdlog::debug("DESTRUCT: system");
}

void system::startup(const bool need_initialize) {
    spdlog::info("startup SLAM system");
    system_is_running_ = true;

    if (!need_initialize) {
        tracker_->tracking_state_ = tracker_state_t::Lost;
    }

    mapping_thread_ = std::unique_ptr<std::thread>(new std::thread(&openvslam::mapping_module::run, mapper_));
}

void system::shutdown() {
    // terminate the other threads
    mapper_->request_terminate();
    // wait until they stop
    while (!mapper_->is_terminated()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // wait until the threads stop
    mapping_thread_->join();

    spdlog::info("shutdown SLAM system");
    system_is_running_ = false;
}

void system::save_keyframe_trajectory(const std::string& path, const std::string& format) const {
    pause_other_threads();
    io::trajectory_io trajectory_io(map_db_);
    trajectory_io.save_keyframe_trajectory(path, format);
    resume_other_threads();
}

const std::shared_ptr<publish::map_publisher> system::get_map_publisher() const {
    return map_publisher_;
}

const std::shared_ptr<publish::frame_publisher> system::get_frame_publisher() const {
    return frame_publisher_;
}


std::shared_ptr<Mat44_t> system::feed_multi_frames(const std::vector<cv::Mat> &imgs, double timestamp,
                                                   const std::vector<cv::Mat> &masks) {

    check_reset_request();

    const auto cam_pose_wc = tracker_->track_multi_images(imgs, timestamp, masks);

    frame_publisher_->update(tracker_);
    if (tracker_->tracking_state_ == tracker_state_t::Tracking && cam_pose_wc) {
        map_publisher_->set_current_cam_pose(tracker_->curr_frm_.get_cam_pose());
        map_publisher_->set_current_cam_pose_wc(*cam_pose_wc);
    }

    return cam_pose_wc;
}

void system::pause_tracker() {
    tracker_->request_pause();
}

bool system::tracker_is_paused() const {
    return tracker_->is_paused();
}

void system::resume_tracker() {
    tracker_->resume();
}

void system::request_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
}

bool system::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void system::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool system::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void system::check_reset_request() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    if (reset_is_requested_) {
        tracker_->reset();
        reset_is_requested_ = false;
    }
}

void system::pause_other_threads() const {
    // pause the mapping module
    if (mapper_ && !mapper_->is_terminated()) {
        mapper_->request_pause();
        while (!mapper_->is_paused() && !mapper_->is_terminated()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    }
}

void system::resume_other_threads() const {
    // resume the mapping module
    if (mapper_) {
        mapper_->resume();
    }
}

} // namespace openvslam
