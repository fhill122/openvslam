/*
 * Created by Ivan B on 2022/10/21.
 */

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include "openvslam/config.h"

#include "ivtb/log.h"

using namespace std;

int main(int argc, char* argv[]){
    spdlog::set_level(spdlog::level::debug);

    AssertLog(argc==2, "input config yaml file");
    openvslam::config config(argv[1]);
    int n = 0;
    for (const auto &overlap : config.cam_rig_->overlaps) {
        n += overlap.size();
    }
    spdlog::info("loaded {} overlaps", n);

    auto& overlaps = config.cam_rig_->overlaps;
    for (int i=0; i<overlaps.size(); ++i){
        for (int j=0; j<overlaps[i].size(); ++j){
            // spdlog::debug("{} have {} overlap(s)", i, overlaps[i].size());
            const openvslam::camera::CamOverlap &overlap = overlaps[i][j];
            string file = to_string(overlap.ind1) + "_" + to_string(overlap.ind2) + ".png";
            cv::imwrite(file, overlap.mask);
            spdlog::debug("writing mask {}", file);
        }
    }

    return 0;
}

