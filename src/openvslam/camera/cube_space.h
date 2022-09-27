/*
 * Created by Ivan B on 2022/8/4.
 */

#ifndef SRC_OPENVSLAM_CAMERA_CUBE_SPACE_H_
#define SRC_OPENVSLAM_CAMERA_CUBE_SPACE_H_

#include "base.h"
#include "ivtb/log.h"

#include <Eigen/Dense>

using namespace std;

namespace openvslam {
namespace camera {

// todo ivan. reconsider if use it as a camera model
class CubeSpace final : public base{
// static
public:
    enum Face{
        UNKNOWN_FACE = -1,
        FRONT_FACE = 0,
        LEFT_FACE = 1,
        RIGHT_FACE = 2,
        UPPER_FACE = 3,
        LOWER_FACE = 4,
        BACK_FACE = 5
    };

    struct CubePoint{
        Face face = UNKNOWN_FACE;
        float u = -1;
        float v = -1;
        CubePoint() = default;
        CubePoint(Face face, float u, float v):face(face), u(u), v(v){};
    };

    static constexpr int kCoordEnc = 10000;  // apply to u only

    static inline std::tuple<Face,float,float> ToFaceCoord(float u, float v){
        int face = u/kCoordEnc;
        float u_out = u - face*kCoordEnc;
        return {(Face)face, u, v};
    }

    static inline std::tuple<float,float> ToCompactCoord(Face face, float u, float v){
        return {((int)face)*kCoordEnc+u, v};
    }

    template<class F>
    static void RotationToFace(Face face, Eigen::Matrix<F,3,3> &R_local);

    template<class F>
    static Face GetFace(F x, F y, F z);

private:

public:
    std::unique_ptr<base> pimpl_;
    const double f_;
    const double c_;
    const double f_inv_;
    const double size_;

public:
    CubeSpace(const std::string& name, setup_type_t setup_type, const color_order_t& color_order,
              const unsigned int size, const double fps,
              const double baseline, std::unique_ptr<base> pimpl);

    CubeSpace(const YAML::Node& yaml_node, std::unique_ptr<base> pimpl);

    // convert bearing in rig frame to image at given face
    cv::Point2f convert_bearing_to_point(const Vec3_t& bearing, const Face face) const;

    CubePoint convert_point_to_cube(const cv::Point2f& undist_pt) const ;

    void convert_bearings_to_cube(const eigen_alloc_vector<Vec3_t>& bearings,
                                  std::vector<CubePoint>& cube_keypts) const;

    // overrides
    void show_parameters() const override;
    image_bounds compute_image_bounds() const override;
    cv::Point2f undistort_point(const cv::Point2f& dist_pt) const override;
    Vec3_t convert_point_to_bearing(const cv::Point2f& undist_pt) const override;
    cv::Point2f convert_bearing_to_point(const Vec3_t& bearing) const override;
    bool reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const override;
    bool reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const override;
    nlohmann::json to_json() const override;
};


/* template implementations */
template<class F>
void CubeSpace::RotationToFace(CubeSpace::Face face, Eigen::Matrix<F, 3, 3>& R_local) {
    // rotation from camera frame to face frame: P_in_face = R_local*P_in_cam
    switch(face) {
        case FRONT_FACE:
            R_local <<
                1, 0, 0,
                0, 1, 0,
                0, 0, 1;
            break;
        case LEFT_FACE:
            R_local <<
                0, 0, 1,
                0, 1, 0,
                -1, 0, 0;
            break;
        case RIGHT_FACE:
            R_local <<
                0, 0,-1,
                0, 1, 0,
                1, 0, 0;
            break;
        case LOWER_FACE:
            R_local <<
                1, 0, 0,
                0, 0,-1,
                0, 1, 0;
            break;
        case UPPER_FACE:
            R_local <<
                1, 0, 0,
                0, 0, 1,
                0,-1, 0;
            break;
        case BACK_FACE:
            R_local <<
                1, 0, 0,
                0,-1, 0,
                0, 0,-1;
            break;
        default:
            R_local <<
                0, 0, 0,
                0, 0, 0,
                0, 0, 0;
            abort();
    }
}

template<class F>
CubeSpace::Face CubeSpace::GetFace(F x, F y, F z) {
    static_assert(is_floating_point<F>::value, "");

    //choose different face according to (x, y, z)
    if(z > 0 && abs(x/z) <= 1 && abs(y/z) <= 1)
        return FRONT_FACE;

    if(x > 0 && abs(y/x) <= 1 && abs(z/x) <= 1)
        return RIGHT_FACE;

    if(x < 0 && abs(y/x) <= 1 && abs(z/x) <= 1)
        return LEFT_FACE;

    if(y > 0 && abs(x/y) <= 1 && abs(z/y) <= 1)
        return LOWER_FACE;

    if(y < 0 && abs(x/y) <= 1 && abs(z/y) <= 1)
        return UPPER_FACE;

    if(z < 0 && abs(x/z) <= 1 && abs(y/z) <= 1)
        return BACK_FACE;

    AssertLog(false, "should never reach here");
    return UNKNOWN_FACE;
}

}}
#endif // SRC_OPENVSLAM_CAMERA_CUBE_SPACE_H_
