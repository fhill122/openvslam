/**
* This file is part of CubemapSLAM.
*
* Copyright (C) 2017-2019 Yahui Wang <nkwangyh at mail dot nankai dot edu dot cn> (Nankai University)
* For more information see <https://github.com/nkwangyh/CubemapSLAM>
*
* CubemapSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CubemapSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CubemapSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

/*
* CubemapSLAM is based on ORB-SLAM2 and Multicol-SLAM which were also released under GPLv3
* For more information see <https://github.com/raulmur/ORB_SLAM2>
* Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* and <https://github.com/urbste/MultiCol-SLAM>
* Steffen Urban <urbste at googlemail.com>
*/

#include "g2o_cubemap_vertices_edges.h"

namespace g2o {

//Multi-pinhole Only Pose
bool EdgeSE3ProjectXYZMultiPinholeOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZMultiPinholeOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZMultiPinholeOnlyPose::linearizeOplus() {
    auto * vi = static_cast<shot_vertex *>(_vertices[0]);
    Vector3d p_body = vi->estimate().map(Xw);
    double px = p_body.x();
    double py = p_body.y();
    double pz = p_body.z();

    //get cube face xyz_trans reside on
    Eigen::Matrix<double,3,3> R_fc;
    CubeSpace::RotationToFace(_face, R_fc);

    // dP / dXi
    Eigen::Matrix<double,3,6> dRigPt_dXi;
    // note ivan. this order as rotation then translation, check slam book p164 7.44. same as orbslam3 SE3deriv
    dRigPt_dXi << 0,  pz, -py, 1, 0, 0,
                -pz,   0,  px, 0, 1, 0,
                 py, -px,   0, 0, 0, 1;

    // de / dP. partial derivative of  projection err over local point
    Eigen::Matrix4d T_fc = Eigen::Matrix4d::Identity(); // transform from camera to face
    T_fc.block<3,3>(0,0) = R_fc;
    Eigen::Matrix4d T_fb = T_fc * pose_cb_;  // body to face
    Vector3d localPt = T_fb.block<3,3>(0,0) * p_body + T_fb.block<3,1>(0,3);
    Eigen::Matrix<double,2,3> de_dLocalPt;
    de_dLocalPt << fx/localPt[2], 0, -fx*localPt[0]/(localPt[2]*localPt[2]),
                   0, fy/localPt[2], -fy*localPt[1]/(localPt[2]*localPt[2]);

    _jacobianOplusXi = -1.0 * de_dLocalPt * T_fb.block<3,3>(0,0) * dRigPt_dXi;
}

Vector2d EdgeSE3ProjectXYZMultiPinholeOnlyPose::multipinhole_project(const Vector3d & trans_xyz) const{
    auto point = _cam->convert_bearing_to_point(trans_xyz,_face);
    return Vector2d(point.x, point.y);
}

Vector2d EdgeSE3ProjectXYZMultiPinholeOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

//Multi-pinhole localBA
EdgeSE3ProjectXYZMultiPinhole::EdgeSE3ProjectXYZMultiPinhole() : BaseBinaryEdge<2, Vector2d, landmark_vertex, shot_vertex>() {}

bool EdgeSE3ProjectXYZMultiPinhole::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZMultiPinhole::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZMultiPinhole::linearizeOplus() {
    auto * vj = static_cast<shot_vertex *>(_vertices[1]);
    SE3Quat T_bw(vj->estimate());
    auto* vi = static_cast<landmark_vertex*>(_vertices[0]);
    Vector3d p_world = vi->estimate();
    Vector3d p_body = T_bw.map(p_world);
    double px = p_body.x();
    double py = p_body.y();
    double pz = p_body.z();

    //get cube face xyz_trans residex on
    Eigen::Matrix<double,3,3> R_fc;
    CubeSpace::RotationToFace(_face, R_fc);

    // dP / dXi
    Eigen::Matrix<double,3,6> dRigPt_dXj;
    // note ivan. this order as rotation then translation, check slam book p164 7.44. same as orbslam3 SE3deriv
    dRigPt_dXj << 0,  pz, -py, 1, 0, 0,
                -pz,   0,  px, 0, 1, 0,
                 py, -px,   0, 0, 0, 1;

    // de / dP. partial derivative of  projection err over local point
    Eigen::Matrix4d T_fc = Eigen::Matrix4d::Identity(); // transform from camera to face
    T_fc.block<3,3>(0,0) = R_fc;
    Eigen::Matrix4d T_fb = T_fc * pose_cb_;  // body to face
    Vector3d localPt = T_fb.block<3,3>(0,0) * p_body + T_fb.block<3,1>(0,3);
    Eigen::Matrix<double,2,3> de_dLocalPt;
    de_dLocalPt << fx/localPt[2], 0, -fx*localPt[0]/(localPt[2]*localPt[2]),
                   0, fy/localPt[2], -fy*localPt[1]/(localPt[2]*localPt[2]);

    _jacobianOplusXj = -1.0 * de_dLocalPt * T_fb.block<3,3>(0,0) * dRigPt_dXj;
    _jacobianOplusXi = -1.0 * de_dLocalPt * T_fb.block<3,3>(0,0) * T_bw.rotation().toRotationMatrix();
}

Vector2d EdgeSE3ProjectXYZMultiPinhole::multipinhole_project(const Vector3d & trans_xyz) const{
    auto point = _cam->convert_bearing_to_point(trans_xyz,_face);
    return Vector2d(point.x, point.y);
}

Vector2d EdgeSE3ProjectXYZMultiPinhole::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

} //NAMESPACE g2o
