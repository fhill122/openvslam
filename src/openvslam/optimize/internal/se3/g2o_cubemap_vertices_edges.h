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

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/eigen_types.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "openvslam/camera/cube_space.h"
#include "openvslam/type.h"
#include "openvslam/optimize/internal/landmark_vertex.h"
#include "openvslam/optimize/internal/se3/shot_vertex.h"

namespace g2o {

// todo ivan. fix this ugly thing
using namespace Eigen;
using landmark_vertex = openvslam::optimize::internal::landmark_vertex;
using shot_vertex = openvslam::optimize::internal::se3::shot_vertex;

using CubeSpace = openvslam::camera::CubeSpace;

inline Vector2d project2d(const Vector3d& v){
    Vector2d res;
    res(0) = v(0)/v(2);
    res(1) = v(1)/v(2);
    return res;
}

// Multi-pinhole minimization
class  EdgeSE3ProjectXYZMultiPinholeOnlyPose: public  BaseUnaryEdge<2, Vector2d, shot_vertex>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZMultiPinholeOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const auto* v1 = static_cast<const shot_vertex*>(_vertices[0]);
    if(_face == CubeSpace::UNKNOWN_FACE){
        _error = Vector2d(0.0, 0.0);
        std::cout << "@func: PoseOptimization computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
        
    _error = _measurementInFace-multipinhole_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
      // note ivan. change to judge on depth in face instead of in rig
    const auto* v1 = static_cast<const shot_vertex*>(_vertices[0]);
    Eigen::Matrix<double,3,3> R_local;
    CubeSpace::RotationToFace(_face, R_local);
    auto p_rig = v1->estimate().map(Xw);
    auto p_face = R_local*p_rig;
    return p_face(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
  Vector2d cam_project(const Vector3d & trans_xyz) const;


  // landmark 3d position in world
  Vector3d Xw;


  void setFace(const CubeSpace::Face& face) { _face = face; }
  void setCam(const CubeSpace *cam) {
      _cam = cam;
      fx = cam->f_;
      fy = cam->f_;
      cx = cam->c_;
      cy = cam->c_;
  }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}
  void setPoseBC(const Eigen::Matrix4d &pose_bc){
      pose_bc_=pose_bc;
      pose_cb_=pose_bc_.inverse();
  }
  void setPoseCB(const Eigen::Matrix4d &pose_cb){
      pose_cb_=pose_cb;
      pose_bc_=pose_cb_.inverse();
  }

private:
  CubeSpace::Face _face;
  const CubeSpace* _cam;
  double fx, fy, cx, cy;
  // cube/camera pose in body frame
  Eigen::Matrix4d pose_bc_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d pose_cb_ = Eigen::Matrix4d::Identity();
  Measurement _measurementInFace;
};

//Multi-pinhole localBA
class  EdgeSE3ProjectXYZMultiPinhole: public  BaseBinaryEdge<2, Vector2d, landmark_vertex, shot_vertex>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZMultiPinhole();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const auto* v1 = static_cast<const shot_vertex*>(_vertices[1]);
    const auto* v2 = static_cast<const landmark_vertex*>(_vertices[0]);
    if(_face == CubeSpace::UNKNOWN_FACE){
        _error = Vector2d(0.0, 0.0);
        std::cout << "@func: BundleAdjustment computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
        
    _error = _measurementInFace-multipinhole_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    // note ivan. change to judge on depth in face instead of in rig
    const auto* v1 = static_cast<const shot_vertex*>(_vertices[1]);
    const auto* v2 = static_cast<const landmark_vertex*>(_vertices[0]);
    Eigen::Matrix<double,3,3> R_local;
    CubeSpace::RotationToFace(_face, R_local);
    auto p_rig = v1->estimate().map(v2->estimate());
    auto p_face = R_local*p_rig;
    return p_face(2)>0.0;
  }
    

  virtual void linearizeOplus();

  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
  Vector2d cam_project(const Vector3d & trans_xyz) const;


  void setCam(const CubeSpace *cam) {
      _cam = cam;
      fx = cam->f_;
      fy = cam->f_;
      cx = cam->c_;
      cy = cam->c_;
  }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}
  void setFace(const CubeSpace::Face& face) { _face = face; }
  void setPoseBC(const Eigen::Matrix4d &pose_bc){
      pose_bc_=pose_bc;
      pose_cb_=pose_bc_.inverse();
  }
  void setPoseCB(const Eigen::Matrix4d &pose_cb){
      pose_cb_=pose_cb;
      pose_bc_=pose_cb_.inverse();
  }

private:
  double fx, fy, cx, cy;
  CubeSpace::Face _face;
  const CubeSpace* _cam;
  // cube/camera pose in body frame
  Eigen::Matrix4d pose_bc_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d pose_cb_ = Eigen::Matrix4d::Identity();
  Measurement _measurementInFace;
};

} //NAMESPACE g2o
