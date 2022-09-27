#include "openvslam/config.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/radial_division.h"
#include "openvslam/camera/cube_space.h"
#include "openvslam/util/string.h"

#include <iostream>
#include <memory>

#include <ivtb/log.h>
#include <ivtb/easy_yaml.h>
#include <spdlog/spdlog.h>

using namespace std;

namespace openvslam {

config::config(const std::string& config_file_path)
    : config(YAML::LoadFile(config_file_path), config_file_path) {}

config::config(const YAML::Node& yaml_node, const std::string& config_file_path)
    : config_file_path_(config_file_path), yaml_node_(yaml_node) {
    spdlog::debug("CONSTRUCT: config");

    spdlog::info("config file loaded: {}", config_file_path_);

    // camera_ = CamFromYaml(yaml_node_["Camera"]);
    //
    // if (camera_->setup_type_ == camera::setup_type_t::Stereo || camera_->setup_type_ == camera::setup_type_t::RGBD) {
    //     if (camera_->model_type_ == camera::model_type_t::Equirectangular) {
    //         throw std::runtime_error("Not implemented: Stereo or RGBD of equirectangular camera model");
    //     }
    // }

    cam_rig_ = CamRigFromYaml(yaml_node_);
}

config::~config() {
    // delete camera_;
    // camera_ = nullptr;

    spdlog::debug("DESTRUCT: config");
}

std::ostream& operator<<(std::ostream& os, const config& cfg) {
    os << cfg.yaml_node_;
    return os;
}

camera::base* config::CamFromYaml(const YAML::Node& yaml_node) {
    auto normal_cam_from_yaml =
        [](camera::model_type_t camera_model_type, const YAML::Node& base_node) -> camera::base*{
        switch (camera_model_type) {
            case camera::model_type_t::Perspective: {
                return new camera::perspective(base_node);
            }
            case camera::model_type_t::Fisheye: {
                return new camera::fisheye(base_node);
            }
            case camera::model_type_t::Equirectangular: {
                return new camera::equirectangular(base_node);
            }
            case camera::model_type_t::RadialDivision: {
                return new camera::radial_division(base_node);
            }
            default:
                AssertLog(false, "not implemented");
        }
    };

    spdlog::debug("load camera model type");
    const auto camera_model_type = camera::base::load_model_type(yaml_node);
    camera::base* camera;

    spdlog::debug("load camera model parameters");
    try {
        if (camera_model_type == camera::model_type_t::VirtualCube){
            const auto base_model = camera::base::load_model_type(yaml_node["base_camera"]);
            auto base_cam = normal_cam_from_yaml(base_model, yaml_node["base_camera"]);
            camera = new camera::CubeSpace(yaml_node, unique_ptr<camera::base>(base_cam));
        } else{
            camera = normal_cam_from_yaml(camera_model_type, yaml_node);
        }
    }
    catch (const std::exception& e) {
        spdlog::error("failed in loading camera model parameters: {}", e.what());
        delete camera;
        camera = nullptr;
        throw;
    }
    return camera;
}

unique_ptr<camera::CameraRig> config::CamRigFromYaml(const YAML::Node& yaml_node) {
    unique_ptr<camera::CameraRig> rig = unique_ptr<camera::CameraRig>(new camera::CameraRig());

    struct CameraRig{
        unsigned int num_cameras;
        vector<vector<double>> poses;
    } CameraRig;
    SET_FROM_YAML(yaml_node, CameraRig.num_cameras);
    SET_FROM_YAML(yaml_node, CameraRig.poses);
    AssertLog(CameraRig.num_cameras == CameraRig.poses.size(), "");
    AssertLog(CameraRig.num_cameras>=1, "at least one camera");


    rig->cameras.resize(CameraRig.num_cameras);
    rig->poses_inv.resize(CameraRig.num_cameras);
    rig->poses.resize(CameraRig.num_cameras);
    for (unsigned int i = 0; i < CameraRig.num_cameras; ++i) {
        vector<double> pose = CameraRig.poses[i];
        AssertLog(pose.size()==7, "");
        rig->poses_inv[i].setIdentity();
        rig->poses_inv[i].block<3,3>(0,0) = Eigen::Quaterniond(pose[0], pose[1],
                                                           pose[2], pose[3]).toRotationMatrix();
        rig->poses_inv[i](0,3) = pose[4];
        rig->poses_inv[i](1,3) = pose[5];
        rig->poses_inv[i](2,3) = pose[6];

        rig->poses[i] = rig->poses_inv[i].inverse();

        YAML::Node cam_node;
        if (i==0){
            cam_node = yaml_node["Camera"];
        } else{
            cam_node = yaml_node["Camera" + to_string(i)];
        }
        AssertLog(cam_node, "");
        rig->cameras[i] = unique_ptr<camera::base>(CamFromYaml(cam_node));
    }

    return rig;
}

} // namespace openvslam
