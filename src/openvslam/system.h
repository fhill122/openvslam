#ifndef OPENVSLAM_SYSTEM_H
#define OPENVSLAM_SYSTEM_H

#include "openvslam/type.h"
#include "openvslam/data/bow_vocabulary_fwd.h"
#include "openvslam/camera/camera_rig.h"

#include <string>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <memory>

#include <opencv2/core/core.hpp>

namespace openvslam {

class config;
class tracking_module;
class mapping_module;

namespace camera {
class base;
} // namespace camera

namespace data {
class camera_database;
class map_database;
class bow_database;
} // namespace data

namespace publish {
class map_publisher;
class frame_publisher;
} // namespace publish

class system {
public:
    //! Constructor
    system(const std::shared_ptr<config>& cfg, const std::string& vocab_file_path);

    //! Destructor
    ~system();

    //-----------------------------------------
    // system startup and shutdown

    //! Startup the SLAM system
    void startup(const bool need_initialize = true);

    //! Shutdown the SLAM system
    void shutdown();

    //-----------------------------------------
    // data I/O

    //! Save the frame trajectory in the specified format
    void save_frame_trajectory(const std::string& path, const std::string& format) const;

    //! Save the keyframe trajectory in the specified format
    void save_keyframe_trajectory(const std::string& path, const std::string& format) const;

    //! Get the map publisher
    const std::shared_ptr<publish::map_publisher> get_map_publisher() const;

    //! Get the frame publisher
    const std::shared_ptr<publish::frame_publisher> get_frame_publisher() const;

    //-----------------------------------------
    // module management

    //! Enable the mapping module
    void enable_mapping_module();

    //! Disable the mapping module
    void disable_mapping_module();

    //! The mapping module is enabled or not
    bool mapping_module_is_enabled() const;

    //-----------------------------------------
    // data feeding methods

    //! Feed a monocular frame to SLAM system
    //! (NOTE: distorted images are acceptable if calibrated)
    std::shared_ptr<Mat44_t> feed_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //! Feed a stereo frame to SLAM system
    //! (Note: Left and Right images must be stereo-rectified)
    std::shared_ptr<Mat44_t> feed_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //! Feed an RGBD frame to SLAM system
    //! (Note: RGB and Depth images must be aligned)
    std::shared_ptr<Mat44_t> feed_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //-----------------------------------------
    // management for pause

    //! Pause the tracking module
    void pause_tracker();

    //! The tracking module is paused or not
    bool tracker_is_paused() const;

    //! Resume the tracking module
    void resume_tracker();

    //-----------------------------------------
    // management for reset

    //! Request to reset the system
    void request_reset();

    //! Reset of the system is requested or not
    bool reset_is_requested() const;

    //-----------------------------------------
    // management for terminate

    //! Request to terminate the system
    void request_terminate();

    //!! Termination of the system is requested or not
    bool terminate_is_requested() const;

private:
    //! Check reset request of the system
    void check_reset_request();

    //! Pause the mapping module
    void pause_other_threads() const;

    //! Resume the mapping module
    void resume_other_threads() const;

    //! config
    const std::shared_ptr<config> cfg_;

    //! map database
    data::map_database* map_db_ = nullptr;

    //! camera rig
    camera::CameraRig *cam_rig_ = nullptr;

    //! BoW vocabulary
    data::bow_vocabulary* bow_vocab_ = nullptr;

    //! tracker
    tracking_module* tracker_ = nullptr;

    //! mapping module
    mapping_module* mapper_ = nullptr;
    //! mapping thread
    std::unique_ptr<std::thread> mapping_thread_ = nullptr;

    //! frame publisher
    std::shared_ptr<publish::frame_publisher> frame_publisher_ = nullptr;
    //! map publisher
    std::shared_ptr<publish::map_publisher> map_publisher_ = nullptr;

    //! system running status flag
    std::atomic<bool> system_is_running_{false};

    //! mutex for reset flag
    mutable std::mutex mtx_reset_;
    //! reset flag
    bool reset_is_requested_ = false;

    //! mutex for terminate flag
    mutable std::mutex mtx_terminate_;
    //! terminate flag
    bool terminate_is_requested_ = false;

    //! mutex for flags of enable/disable mapping module
    mutable std::mutex mtx_mapping_;

    //! mutex for flags of enable/disable loop detector
    mutable std::mutex mtx_loop_detector_;
};

} // namespace openvslam

#endif // OPENVSLAM_SYSTEM_H
