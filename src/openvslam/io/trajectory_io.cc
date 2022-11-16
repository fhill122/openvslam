#include "openvslam/data/multi_keyframe.h"
#include "openvslam/data/map_database.h"
#include "openvslam/io/trajectory_io.h"

#include <iostream>
#include <iomanip>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <fstream>

namespace openvslam {
namespace io {

trajectory_io::trajectory_io(data::map_database* map_db)
    : map_db_(map_db) {}

void trajectory_io::save_keyframe_trajectory(const std::string& path, const std::string& format) const {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // 1. acquire keyframes and sort them

    assert(map_db_);
    auto keyfrms = map_db_->get_all_keyframes();
    std::sort(keyfrms.begin(), keyfrms.end(),
              [&](const std::shared_ptr<data::MultiKeyframe>& keyfrm_1,
                  const std::shared_ptr<data::MultiKeyframe>& keyfrm_2) {
        return *keyfrm_1 < *keyfrm_2;
    });

    // 2. save the keyframes

    if (keyfrms.empty()) {
        spdlog::warn("there are no valid keyframes, cannot dump keyframe trajectory");
        return;
    }

    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open()) {
        spdlog::critical("cannot create a file at {}", path);
        throw std::runtime_error("cannot create a file at " + path);
    }

    spdlog::info("dump keyframe trajectory in \"{}\" format from keyframe {} to keyframe {} ({} keyframes)",
                 format, (*keyfrms.begin())->id_, (*keyfrms.rbegin())->id_, keyfrms.size());

    for (const auto& keyfrm : keyfrms) {
        const Mat44_t cam_pose_wc = keyfrm->getCamPoseInv();
        const auto timestamp = keyfrm->timestamp_;

        if (format == "KITTI") {
            ofs << std::setprecision(9)
                << cam_pose_wc(0, 0) << " " << cam_pose_wc(0, 1) << " " << cam_pose_wc(0, 2) << " " << cam_pose_wc(0, 3) << " "
                << cam_pose_wc(1, 0) << " " << cam_pose_wc(1, 1) << " " << cam_pose_wc(1, 2) << " " << cam_pose_wc(1, 3) << " "
                << cam_pose_wc(2, 0) << " " << cam_pose_wc(2, 1) << " " << cam_pose_wc(2, 2) << " " << cam_pose_wc(2, 3) << std::endl;
        }
        else if (format == "TUM") {
            const Mat33_t& rot_wc = cam_pose_wc.block<3, 3>(0, 0);
            const Vec3_t& trans_wc = cam_pose_wc.block<3, 1>(0, 3);
            const Quat_t quat_wc = Quat_t(rot_wc);
            ofs << std::setprecision(15)
                << timestamp << " "
                << std::setprecision(9)
                << trans_wc(0) << " " << trans_wc(1) << " " << trans_wc(2) << " "
                << quat_wc.x() << " " << quat_wc.y() << " " << quat_wc.z() << " " << quat_wc.w() << std::endl;
        }
        else {
            throw std::runtime_error("Not implemented: trajectory format \"" + format + "\"");
        }
    }

    ofs.close();
}

} // namespace io
} // namespace openvslam
