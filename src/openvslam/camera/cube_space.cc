/*
 * Created by Ivan B on 2022/8/4.
 */

#include "cube_space.h"
#include <nlohmann/json.hpp>
#include <ivtb/log.h>

using namespace std;

namespace openvslam {
namespace camera {



CubeSpace::CubeSpace(const string& name, setup_type_t setup_type, const color_order_t& color_order,
                     const unsigned int size, const double fps,
                     const double baseline, std::unique_ptr<base> pimpl):
        base(name, setup_type, model_type_t::VirtualCube, color_order, pimpl->cols_, pimpl->rows_, fps,
           0.5*size*baseline, baseline), pimpl_(move(pimpl)),
        f_(0.5*size), c_(0.5*size), f_inv_(1.0/f_), size_(size){
    img_bounds_ = compute_image_bounds();

    inv_cell_width_ = static_cast<double>(num_grid_cols_) / (img_bounds_.max_x_ - img_bounds_.min_x_);
    inv_cell_height_ = static_cast<double>(num_grid_rows_) / (img_bounds_.max_y_ - img_bounds_.min_y_);
}

CubeSpace::CubeSpace(const YAML::Node& yaml_node, std::unique_ptr<base> pimpl) :
      CubeSpace(yaml_node["name"].as<string>(),
                load_setup_type(yaml_node),
                load_color_order(yaml_node),
                yaml_node["size"].as<unsigned int>(),
                yaml_node["fps"].as<double>(),
                yaml_node["baseline"].as<double>(), move(pimpl)) {}

cv::Point2f CubeSpace::convert_bearing_to_point(const Vec3_t& bearing, const CubeSpace::Face face) const{
    Mat33_t R;
    RotationToFace(face, R);
    Vec3_t bearing_in_face = R*bearing;
    const auto x_normalized = bearing_in_face(0) / bearing_in_face(2);
    const auto y_normalized = bearing_in_face(1) / bearing_in_face(2);
    return cv::Point2f(f_ * x_normalized + c_, f_ * y_normalized + c_);
}

CubeSpace::CubePoint CubeSpace::convert_point_to_cube(const cv::Point2f& undist_pt) const {
    Vec3_t bearing = convert_point_to_bearing(undist_pt);
    Face face = GetFace(bearing.x(), bearing.y(), bearing.z());
    auto face_point = convert_bearing_to_point(bearing, face);
    return CubePoint(face, face_point.x, face_point.y);
}

void CubeSpace::convert_bearings_to_cube(const eigen_alloc_vector<Vec3_t>& bearings,
                                         vector<CubePoint> &cube_keypts) const {
    cube_keypts.clear();
    cube_keypts.reserve(bearings.size());
    for (const Vec3_t& bearing : bearings) {
        Face face = GetFace(bearing.x(), bearing.y(), bearing.z());
        auto face_point = convert_bearing_to_point(bearing, face);
        cube_keypts.emplace_back(face, face_point.x, face_point.y);
    }
}

void CubeSpace::show_parameters() const {
    show_common_parameters();
    std::cout << "base camera: " << std::endl;
    pimpl_->show_parameters();
}

image_bounds CubeSpace::compute_image_bounds() const {
    // is it too much of a hassle to implement the cubespace as a camera model?
    // AssertLog(false, "not implemented");

    return pimpl_->compute_image_bounds();
}

cv::Point2f CubeSpace::undistort_point(const cv::Point2f& dist_pt) const {
    return pimpl_->undistort_point(dist_pt);
}

Vec3_t CubeSpace::convert_point_to_bearing(const cv::Point2f& undist_pt) const {
    return pimpl_->convert_point_to_bearing(undist_pt);
}

cv::Point2f CubeSpace::convert_bearing_to_point(const Vec3_t& bearing) const {
    return pimpl_->convert_bearing_to_point(bearing);
}

bool CubeSpace::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    return pimpl_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);
}

bool CubeSpace::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    return pimpl_->reproject_to_bearing(rot_cw, trans_cw, pos_w, reproj);
}

nlohmann::json CubeSpace::to_json() const {
    // todo ivan.
    AssertLog(false, "not implemented");
    return {};
}


}}