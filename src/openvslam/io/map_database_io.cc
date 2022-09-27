#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/io/map_database_io.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <fstream>

namespace openvslam {
namespace io {

map_database_io::map_database_io(data::camera_database* cam_db, data::map_database* map_db,
                                 data::bow_database* bow_db, data::bow_vocabulary* bow_vocab)
    : cam_db_(cam_db), map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab) {}


} // namespace io
} // namespace openvslam
