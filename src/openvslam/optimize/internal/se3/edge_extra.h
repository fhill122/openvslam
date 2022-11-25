/*
 * Created by Ivan B on 2022/10/20.
 */

#ifndef OPENVSLAM_SRC_OPENVSLAM_OPTIMIZE_INTERNAL_SE3_EDGE_EXTRA_H_
#define OPENVSLAM_SRC_OPENVSLAM_OPTIMIZE_INTERNAL_SE3_EDGE_EXTRA_H_

#include "openvslam/camera/cube_space.h"

namespace openvslam {

namespace optimize {
namespace internal {
namespace se3 {

struct CubeSpaceExtra {
    camera::CubeSpace::Face face;
    const Eigen::Matrix4d& T_B_C;

    CubeSpaceExtra(camera::CubeSpace::Face face, const Eigen::Matrix4d& T_B_C)
        : face(face), T_B_C(T_B_C) {}
};

}}}}
#endif // OPENVSLAM_SRC_OPENVSLAM_OPTIMIZE_INTERNAL_SE3_EDGE_EXTRA_H_
