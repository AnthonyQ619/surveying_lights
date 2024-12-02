#define _USE_MATH_DEFINES


#include "triangulation_projective_geo.h"
#include "geospatial.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include "rotation.h"


using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixXf;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;
using Eigen::JacobiSVD;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::VectorXd;

double TPGeo::sinc(double x) {
    if (x == 0.0) {
        return 1;
    }
    else{
        double v = sin(x) / x;
        return v;
    }
}
Matrix3d TPGeo::skew(Vector3d w) {
    Matrix3d skewM = Matrix3d::Zero();

    skewM(0, 1) = -w(2, 0);
    skewM(0, 2) = w(1, 0);
    skewM(1, 0) = w(2, 0);
    skewM(1, 2) = -w(0, 0);
    skewM(2, 0) = -w(1, 0);
    skewM(2, 1) = w(0, 0);
    
    return skewM;
}
Matrix3d TPGeo::deparameterize_Omega(Vector3d w) {
    double mag_w = w.norm();
    Matrix3d R = Matrix3d::Identity();

    Matrix3d R1 = skew(w) * sinc(mag_w);

    Matrix3d R2 = ((1 - cos(mag_w)) / (pow(mag_w, 2.0)) * skew(w) * skew(w));

    R = R + R1 + R2;

    return R;
}

//w Must be a 3x1 Vector
void TPGeo::parameterize_Rotation(Matrix3d R, Vector3d *w, double *theta) {
    Matrix3d newR = R - Matrix3d::Identity();
    //Matrix3d b = Matrix3d::Zero();
    //Vector3d V = newR.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    JacobiSVD<MatrixXd> svd(newR, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd V = svd.matrixV().col(2);
    V.transposeInPlace();
    Vector3d V_hat{ { R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1) } };

    double sin_theta = double((V * V_hat)(0, 0) / 2);

    double cos_theta = double((R.trace() - 1) / 2);

    *theta = atan2(sin_theta, cos_theta);

    //std::cout << "BeforeW" << std::endl;
    Vector3d V1 = svd.matrixV().col(2);
    std::cout << "BeforeW" << std::endl;
    *w = (*theta / V1.norm()) * V1;
    std::cout << "After W" << std::endl;

    if (*theta > M_PI) {
        *w = ((1 - 2 * M_PI / *theta) * ceil((*theta - M_PI) / 2 / M_PI)) * (*w);
    }
}

MatrixXd TPGeo::calc_camera_proj_matrix(Matrix3d K, Matrix3d R, VectorXd t) {
    MatrixXd Rnew(R.rows(), R.cols() + t.cols()); // Horizontal Cancatenation
    Rnew << R, t;

    return K * Rnew;
}

MatrixXd TPGeo::homogenize(MatrixXd x) {
    MatrixXd ones = MatrixXd::Ones(1, x.cols());
    MatrixXd xHomog(x.rows() + ones.rows(), x.cols());

    xHomog << x, ones;

    return xHomog;

}

MatrixXd TPGeo::dehomogenize(MatrixXd x) {
    MatrixXd x1Inhomog = x.block(0, 0, x.rows() - 1, x.cols());

    RowVectorXd lastRow = x.block(x.rows() - 1, 0, 1, x.cols());
    std::cout << "X" << std::endl;
    std::cout << x << std::endl;
    std::cout << "inhomogoenous" << std::endl;
    std::cout << x1Inhomog << std::endl;
    std::cout << "last Row" << std::endl;
    std::cout << lastRow << std::endl;

    MatrixXd xinHomog = x1Inhomog.array().rowwise() / lastRow.array();

    return xinHomog;
}
Matrix3d TPGeo::angle_axis_to_rotation(double angle, MatrixXd axis) {
    double c = cos(angle);
    double s = sin(angle);
    double t = 1.0 - c;
    MatrixXd norm_axis = axis / axis.norm();

    double x = norm_axis(0, 0);
    double y = norm_axis(1, 0);
    double z = norm_axis(2, 0);
    Matrix3d R = Matrix3d::Zero();
    R(0, 0) = t * pow(x, 2) + c;
    R(0, 1) = t * x * y - z * s;
    R(0, 2) = t * x * z + y * s;
    R(1, 0) = t * x * y + z * s;
    R(1, 1) = t * pow(y, 2) + c;
    R(1, 2) = t * y * z - x * s;
    R(2, 0) = t * x * z - y * s;
    R(2, 1) = t * y * z + x * s;
    R(2, 2) = t * pow(z, 2) + c;

    return R;
}
Matrix3d TPGeo::interpolate_rotation(double alpha, Matrix3d r1, Matrix3d r2) {
    Vector3d w = Vector3d::Zero();
    double theta = 0;
    r1.transposeInPlace();

    Matrix3d newR = r2 * r1;

    parameterize_Rotation(newR, &w, &theta);

    return deparameterize_Omega(alpha * w) * r1;
}
Vector3d TPGeo::camera_translation(Matrix3d R1, VectorXd t1, Matrix3d R2) {
    R1.transposeInPlace();
    return R2 * (R1 * t1);
}
Matrix3d TPGeo::calc_projective_transformation(Matrix3d K1, Matrix3d R1, Matrix3d K2, Matrix3d R2) {
    R1.transposeInPlace();

    return K2 * R2 * R1 * K1.inverse();
}

void TPGeo::epipolar_rectification(Matrix3d K1, Matrix3d R1, Vector3d t1,
    Matrix3d K2, Matrix3d R2, Vector3d t2,
    Matrix3d *Krectified, Matrix3d *Rrectified, Vector3d *t1rectified,
    Vector3d *t2rectified, Matrix3d *H1, Matrix3d *H2) {

    std::cout << "Inside epipolar_rectification" << std::endl;
    Matrix3d Rinterp = interpolate_rotation(0.5, R1, R2);

    std::cout << Rinterp << std::endl;
    Vector3d u = camera_translation(R2, t2, Rinterp) - camera_translation(R1, t1, Rinterp);
    Vector3d vhat{ {1.0, 0.0, 0.0} };
    
    std::cout << "U" << std::endl;
    std::cout << u << std::endl;
    MatrixXd uT = u.transpose();

    std::cout << "U.T" << std::endl;
    std::cout << uT << std::endl;


    double neg = (uT * vhat)(0,0);

    if (0 > neg) {
        vhat(0, 0) = -1.0;
    }

    double angle = acos((uT * vhat)(0,0) / u.norm());


    Vector3d axis = u.cross(vhat);

    Matrix3d R_X = angle_axis_to_rotation(angle, axis);
    *Rrectified = R_X * Rinterp;

    *t1rectified = camera_translation(R1, t1, *Rrectified);
    *t2rectified = camera_translation(R2, t2, *Rrectified);

    double alpha = (K1(0,0) + K2(0,0) + K1(1,1) + K2(1,1)) / 4;


    double x0 = (K1(0, 2) + K2(0, 2)) / 2;


    double y0 = (K1(1, 2) + K2(1, 2)) / 2;

    *Krectified = Matrix3d::Zero();


    Krectified->coeffRef(0, 0) = alpha;
    Krectified->coeffRef(0, 2) = x0;
    Krectified->coeffRef(1, 1) = alpha;
    Krectified->coeffRef(1, 2) = y0;
    Krectified->coeffRef(2, 2) = 1.0;

    *H1 = calc_projective_transformation(K1, R1, *Krectified, *Rrectified);

    *H2 = calc_projective_transformation(K2, R2, *Krectified, *Rrectified);
}
MatrixXd TPGeo::calc_epipole(MatrixXd E) {
    JacobiSVD<MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd v = svd.matrixV().col(2);

    return v;
}
void TPGeo::backprojection(VectorXd x, MatrixXd P, RowVector4d *row1, RowVector4d *row2) {
    double x1_coeff = P(2, 0)* x(0, 0) - P(0, 0);
    double y1_coeff = P(2, 1) * x(0, 0) - P(0, 1);
    double z1_coeff = P(2, 2) * x(0, 0) - P(0, 2);
    double w1_coeff = P(2, 3) * x(0, 0) - P(0, 3);

    double x2_coeff = P(2, 0) * x(1, 0) - P(1, 0);
    double y2_coeff = P(2, 1) * x(1, 0) - P(1, 1);
    double z2_coeff = P(2, 2) * x(1, 0) - P(1, 2);
    double w2_coeff = P(2, 3) * x(1, 0) - P(1, 3);

    *row1 << x1_coeff, y1_coeff, z1_coeff, w1_coeff;
    *row2 << x2_coeff, y2_coeff, z2_coeff, w2_coeff;
}
MatrixXd TPGeo::triangulation2View(VectorXd x1, VectorXd x2, MatrixXd P1rectified, MatrixXd P2rectified) {
    RowVector4d row1;
    RowVector4d row2;
    RowVector4d row3;
    RowVector4d row4;

    backprojection(x1, P1rectified, &row1, &row2);
    backprojection(x2, P2rectified, &row3, &row4);

    MatrixXd C(row1.rows() + row2.rows() + row3.rows() + row4.rows(), row1.cols());

    C << row1, row2, row3, row4;

    JacobiSVD<MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd v = svd.matrixV().col(3);

    return dehomogenize(v);
}
//Testing functions

void rectified_calibration_matrices(Matrix3d Krectified, Matrix3d H1, Matrix3d H2, 
    const MatrixXf &I1, const MatrixXf &I2, Matrix3d* K1rectified, Matrix3d* K2rectified) {
    //points are defined by[x, y]!!!
    TPGeo tp = TPGeo();

    Vector2d I1_top_left{ {-0.5, -0.5} };
    Vector2d I1_top_right{ {I1.cols() - 0.5, -0.5}};

    Vector2d I1_bottom_left{ {-0.5, I1.rows() - 0.5}};
    Vector2d I1_bottom_right{ {I1.cols() - 0.5, I1.rows() - 0.5}};


    Vector2d I2_top_left{ {-0.5, -0.5} };
    Vector2d I2_top_right{ {I2.cols() - 0.5, -0.5} };

    Vector2d I2_bottom_left{ {-0.5, I2.rows() - 0.5} };
    Vector2d I2_bottom_right{ {I2.cols() - 0.5, I2.rows() - 0.5} };


    MatrixXd pts_I1(I1_top_left.rows(), I1_top_left.cols() + I1_top_right.cols() + I1_bottom_left.cols() + I1_bottom_right.cols());
    pts_I1 << I1_top_left, I1_top_right, I1_bottom_left, I1_bottom_right;

    MatrixXd pts_I2(I2_top_left.rows(), I2_top_left.cols() + I2_top_right.cols() + I2_bottom_left.cols() + I2_bottom_right.cols());
    pts_I2 << I2_top_left, I2_top_right, I2_bottom_left, I2_bottom_right;

    MatrixXd pts_I1_rec = tp.dehomogenize(H1 * tp.homogenize(pts_I1));

    MatrixXd pts_I2_rec = tp.dehomogenize(H2 * tp.homogenize(pts_I2));

    double min_row_I1 = ceil(pts_I1_rec.rowwise().minCoeff()(1, 0));
    double max_row_I1 = ceil(pts_I1_rec.rowwise().maxCoeff()(1, 0));

    double min_row_I2 = ceil(pts_I2_rec.rowwise().minCoeff()(1, 0));
    double max_row_I2 = ceil(pts_I2_rec.rowwise().maxCoeff()(1, 0));

    double min_col_I1 = ceil(pts_I1_rec.rowwise().minCoeff()(0, 0));
    double max_col_I1 = ceil(pts_I1_rec.rowwise().maxCoeff()(0, 0));

    double min_col_I2 = ceil(pts_I2_rec.rowwise().minCoeff()(0, 0));
    double max_col_I2 = ceil(pts_I2_rec.rowwise().maxCoeff()(0, 0));

    Matrix3d T1 = Matrix3d::Identity();
    Matrix3d T2 = Matrix3d::Identity();

    //if add values onto T1[1][2], answer gets closer to Ben's results
    T1(1, 2) = -1*std::min(min_row_I1, min_row_I2) - 0.5;
    T1(0, 2) = -min_col_I1 - 0.5;

    T2(1, 2) = -std::min(min_row_I1, min_row_I2) - 0.5;
    T2(0, 2) = -min_col_I2 - 0.5;

    *K1rectified = T1 * Krectified;

    *K2rectified = T2 * Krectified;
}


MatrixXd geocentric_triangulation2View(const MatrixXf &left_img, const MatrixXf &right_img, MatrixXd x1, MatrixXd x2,
    MatrixXd C, double latitude_origin, double longitude_origin, double plat_yaw, double plat_pitch, 
    double plat_roll, Matrix3d K1, Matrix3d K2, Matrix3d R12, Vector3d t12, bool rectified) {
    
    // Bottom of Phone is "TOP"
    //double gimbal_yaw = M_PI; 
    //double gimbal_pitch = -M_PI / 2;
    //double gimbal_roll = 0;

    //// Top of phone is "top"
    //double gimbal_yaw = 0;
    //double gimbal_pitch = -M_PI / 2;
    //double gimbal_roll = 0;

    //// Top of phone is screen (Laid flat)
    double gimbal_yaw = 0;
    double gimbal_pitch = -M_PI / 4;
    double gimbal_roll = 0;

    //// Left of phone is screen (Laid flat)
    //double gimbal_yaw = -M_PI / 4;
    //double gimbal_pitch = M_PI / 4;
    //double gimbal_roll = 0;

    //// Back of phone is screen (Laid flat)
    //double gimbal_yaw = M_PI / 2;
    //double gimbal_pitch = M_PI / 4;
    //double gimbal_roll = 0;

    //// Right of phone is screen (Laid flat)
    //double gimbal_yaw = -M_PI / 4;
    //double gimbal_pitch = M_PI / 4;
    //double gimbal_roll = 0;

    //// Correct Direction For Gimble on iPhone
    //double gimbal_yaw = -M_PI / 2;
    //double gimbal_pitch = 0;
    //double gimbal_roll = M_PI/2; 

    // Current orientation of IMU Sensor
    //double gimbal_yaw = M_PI / 2;
    //double gimbal_pitch = 0;
    //double gimbal_roll = 0;

    double latitude_radians = latitude_origin * 180.0 / M_PI;
    double longitude_radians = longitude_origin * 180.0 / M_PI;

    double plat_yaw_radians = plat_yaw*180.0 / M_PI; 
    double plat_pitch_radians = plat_pitch*180.0 / M_PI; 
    double plat_roll_radians = plat_roll*180.0 / M_PI; 

    Rotation R = Rotation();

    Matrix3d R1_geocentric = R.rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, plat_pitch_radians, plat_roll_radians, plat_yaw_radians);

    std::cout << "Before t1" << std::endl;
    Vector3d t1_geocentric = -R1_geocentric * C;
    
    Matrix3d R2_geocentric = R12 * R1_geocentric;
    Vector3d t2_geocentric = R12 * t1_geocentric + t12;
    std::cout << "After t2" << std::endl;

    // corresponding rectified :
    TPGeo tp = TPGeo(); 

    MatrixXd P1_geocentric;
    MatrixXd P2_geocentric;

    if (rectified) {
        Matrix3d Krectified;
        Matrix3d Rrectified;
        Vector3d t1rectified;
        Vector3d t2rectified;
        Matrix3d H1; 
        Matrix3d H2; 
   
        std::cout << "Before epipolar_rectification" << std::endl; 
        tp.epipolar_rectification(K1, R1_geocentric, t1_geocentric, K2, R2_geocentric, t2_geocentric, 
            &Krectified, &Rrectified, &t1rectified, &t2rectified, &H1, &H2); 

        std::cout << "After epipolar_rectification" << std::endl; 
        Matrix3d K1rectified;
        Matrix3d K2rectified;  

        rectified_calibration_matrices(Krectified, H1, H2, left_img, right_img, &K1rectified, &K1rectified); 

        P1_geocentric = tp.calc_camera_proj_matrix(K1rectified, Rrectified, t1rectified); 

        P2_geocentric = tp.calc_camera_proj_matrix(K2rectified, Rrectified, t2rectified); 
    }
    else {
        P1_geocentric = tp.calc_camera_proj_matrix(K1, R1_geocentric, t1_geocentric);

        P2_geocentric = tp.calc_camera_proj_matrix(K2, R2_geocentric, t2_geocentric);
    }


    //frobius norm projection matrices
    MatrixXd norm_P1_geocentric = P1_geocentric / P1_geocentric.norm();
    MatrixXd norm_P2_geocentric = P2_geocentric / P2_geocentric.norm();


    MatrixXd triangulated_geocentric_point = tp.triangulation2View(x1, x2, norm_P1_geocentric, norm_P2_geocentric);

    return triangulated_geocentric_point;
}

//int main() {
//    std::string image_path1 = "C:/Users/aquir/Downloads/right0.jpg";
//    cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_GRAYSCALE);
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img12;
//    cv::cv2eigen(img1, img12);
//
//    std::cout << "IMAGE1" << std::endl;
//    std::cout << "Cols: " << img12.cols() << std::endl;
//    std::cout << "Rows: " << img12.rows() << std::endl;
//
//    std::string image_path2 = "C:/Users/aquir/Downloads/left0.jpg";
//    cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_GRAYSCALE);
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img22;
//    cv::cv2eigen(img2, img22);
//
//    std::cout << "IMAGE1" << std::endl;
//    std::cout << "Cols: " << img22.cols() << std::endl;
//    std::cout << "Rows: " << img22.rows() << std::endl;
//
//    MatrixXd x1{ {1862}, {1176} };
//
//    MatrixXd x2{ {1840}, {1176} };
//
//    MatrixXd C{ {-2452517.8384},
//        {-4768302.495884},
//        {3442352.572949} };
//
//        
//    double latitude_origin = 32.875045;
//    double longitude_origin = -117.218468;
//    double plat_yaw = 364.936; 
//    double plat_pitch = -110.848;
//
//    double plat_roll = 1.383;
//    Matrix3d K1{ {3.28387648e+03, 0.00000000e+00, 2.03036278e+03 }, 
//        {0.00000000e+00, 3.29903024e+03, 1.55033756e+03}, 
//        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
//    Matrix3d K2{ {3.26907563e+03, 0.00000000e+00, 2.08651620e+03}, 
//        {0.00000000e+00, 3.27623450e+03, 1.58700571e+03}, 
//        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
//    Matrix3d R12{ {0.99962162,  0.02428445, -0.01291845}, 
//        {-0.02436906, 0.99968233, -0.00643293}, 
//        { 0.01275813, 0.00674531, 0.99989586} };
//    Vector3d t12{ {- 0.33418658, 0.00541115, -0.00189281} };
//
//    MatrixXd returnMat;
//
//    returnMat = geocentric_triangulation2View(img22, img12, x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, 
//        plat_roll, K1, K2, R12, t12);
//
//    std::cout << "Matrix" << std::endl;
//    std::cout << returnMat << std::endl;
//}


// Inhouse Experiment

int main() {
    std::string image_path1 = "C:/Users/Anthony/Downloads/irLeftBackTest1IMU4Str.png";
    cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_GRAYSCALE);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> imgRight; 
    cv::cv2eigen(img1, imgRight); 

    std::cout << "IMAGE1" << std::endl;
    std::cout << "Cols: " << imgRight.cols() << std::endl; 
    std::cout << "Rows: " << imgRight.rows() << std::endl; 

    std::string image_path2 = "C:/Users/Anthony/Downloads/irRightBackTest1IMU4Str.png";
    cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_GRAYSCALE);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> imgLeft;
    cv::cv2eigen(img2, imgLeft);

    std::cout << "IMAGE1" << std::endl;
    std::cout << "Cols: " << imgLeft.cols() << std::endl;
    std::cout << "Rows: " << imgLeft.rows() << std::endl;


    MatrixXd x1{ {343},{200} };
    MatrixXd x2{ {337},{200} };
     
    // Experiment Test with Different Phone position
    // Center 
    double latitude_origin = 32.881169;
    double longitude_origin = -117.234427;

    // Location 0
    //double lonPt = -117.234485;
    //double latPt = 32.881169;

    //// Location 1
    //double lonPt = -117.234454; 
    //double latPt = 32.881193; 

    //// Location 2
    //double lonPt = -117.234455;
    //double latPt = 32.881139;

    //// Location 3
    double lonPt = -117.234355;
    double latPt = 32.881169;


    // Original Location (Warren)
    //double latitude_origin = 32.881168;
    //double longitude_origin = -117.234429; // Was 23

    // Experiment 10 Backwards test (Warren)
    //double latitude_origin = 32.881169;
    //double longitude_origin = -117.234415;
    
    //New Location Experiment Points (CSE)
    // Center Point
    //double latitude_origin = 32.882286;
    //double longitude_origin = -117.234298;

    // Testing Points
    // Short Test
    //double lonPt = -117.234286;
    //double latPt = 32.882280;
    //// Medium Distance
    //double lonPt = -117.234398;
    //double latPt = 32.882325;
    //// Long Distance
    //double lonPt = -117.234398;
    //double latPt = 32.882358;

    //// Experiments 2/3 and 4/5
    //double lonPt = -117.234326;
    //double latPt = 32.882310;

    // Flipped Direction Test
    //double latitude_origin = 32.882299;
    //double longitude_origin = -117.234297;

    //double lonPt = -117.234262;
    //double latPt = 32.882267;

    //Camera Center Point

    double lonRad = longitude_origin * (M_PI / 180); 
    double latRad = latitude_origin * (M_PI / 180); 

    double ellipsoidHeightC = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRad, lonRad, 108); 

    Eigen::Vector3d pts = WGS84GeodeticToWGS84Geocentric(latRad, lonRad, ellipsoidHeightC);  
     

    MatrixXd C{ {pts[0]},
                {pts[1]},
                {pts[2]} };


    std::cout << "Center Point Original" << std::endl; 
    std::cout << std::fixed << C << std::endl; 
    double plat_yaw = 69.818;
    double plat_pitch = -0.746;
    double plat_roll = -1.712;

    // Yaw - Pitch - Roll (In Radians already)
    //double plat_yaw = -1.5465989145492396;
    //double plat_pitch = 0.03573705188246896;
    //double plat_roll = 3.1067147175061356;



    // Realsense Chris's Device Intrinsics
    //Matrix3d K1{ { 382.976928710938, 0.00000000e+00, 315.660247802734 },
    //    {0.00000000e+00,  382.976928710938, 240.120193481445},
    //    {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    //Matrix3d K2{ { 382.976928710938, 0.00000000e+00, 315.660247802734 },
    //    {0.00000000e+00,  382.976928710938, 240.120193481445},
    //    {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    //Matrix3d R12{ {1.0, 0.0, 0.0}, 
    //    {0.0, 1.0, 0.0},
    //    { 0.0, 0.0, 1.0} };
    //Vector3d t12{ { -0.0949900522828102,  0,  0} };
    //bool rectified = false;

    // Realsense Henrik's Device Intrinsics
    // 640x480 resolution Intrinsics
    Matrix3d K1{ { 385.439086914062, 0.00000000e+00, 320.542755126953 },
        {0.00000000e+00,  385.439086914062,  238.801696777344},
        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    Matrix3d K2{ { 385.439086914062, 0.00000000e+00, 320.542755126953 },
        {0.00000000e+00,  385.439086914062,  238.801696777344},
        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    // 1280x720 Resolution Intrinsics
   /* Matrix3d K1{ { 642.398498535156, 0.00000000e+00, 640.904602050781 },
        {0.00000000e+00,  642.398498535156,  358.002838134766},
        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    Matrix3d K2{ { 642.398498535156, 0.00000000e+00, 640.904602050781 },
        {0.00000000e+00,  642.398498535156,  358.002838134766},
        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };*/
    Matrix3d R12{ {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        { 0.0, 0.0, 1.0} };
    Vector3d t12{ {  -0.0949690192937851,  0,  0} };
    bool rectified = false;

    // Stereo Pi Current Setup
    //Matrix3d K1{ {1245.2085834929785, 0.0, 628.1461920126694},
    //             {0.0, 1247.5756375917842, 484.27607116996796},
    //             {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    //Matrix3d K2{ {1242.3555795517973, 0.0, 637.1034091132533},
    //             {0.0, 1245.4676044054588, 507.99521696245586 },
    //             {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
    //Matrix3d R12{ {0.9981716091797479, - 0.019977693908171286, - 0.057046738501198756},
    //              {0.019214181981025133, 0.9997187449075249, - 0.013901305381984053},
    //              {0.05730840983933149, 0.012779781947848153, 0.9982747233778149} };
    //Vector3d t12{ {15.366123963456182, 0.3234755344160923, 0.31904065555118155} };
    //bool rectified = true;

    // Experiment
    MatrixXd returnMat;

    returnMat = geocentric_triangulation2View(imgLeft, imgRight, x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch,
        plat_roll, K1, K2, R12, t12, rectified);

    // Experiment 1
    // 5M point
    //MatrixXd realPt{ {-2453671.634673}, 
    //                 {-4767254.860514}, 
    //                 { 3443028.695045} }; 

    // 10M point
    //MatrixXd realPt{ {-2453676.433921}, 
    //                 {-4767251.916037},
    //                 { 3443029.346818} };

    // 15m Point
    //MatrixXd realPt{ {-2453675.722371},
    //                 {-4767248.693225},
    //                 { 3443034.283022} };

    // Experiment 2

    // Short Test Point of Interest
    //double lonPt = -117.234492;
    //double latPt = 32.881168;

    // Long Test Point of Interest
    //double lonPt = -117.234556; 
    //double latPt = 32.881167; 

    // Experiment 3

    // Short Test
    //double lonPt = -117.234485;
    //double latPt = 32.881168;

    // Long Test
    //double lonPt = -117.234557;
    //double latPt = 32.881168;

    // Experiment 4
    
    // Short Test
    //double lonPt = -117.234482;
    //double latPt = 32.881169;
    
    // Long Test
    //double lonPt = -117.234557;
    //double latPt = 32.881168;

    // Experiment 7/9
    // Short Test Point of Interest
    //double lonPt = -117.234497;
    //double latPt = 32.881168;

    // Long Test Point of Interest
    //double lonPt = -117.234563; 
    //double latPt = 32.881168; 

    // Experiment 10 Backward Test
    //double lonPt = -117.234358;
    //double latPt = 32.881169;

    double lonRadPt = lonPt * (M_PI / 180); 
    double latRadPt = latPt * (M_PI / 180); 

    double ellipsoidHeight = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRadPt, lonRadPt, 108);

    Eigen::Vector3d realPts = WGS84GeodeticToWGS84Geocentric(latRadPt, lonRadPt, ellipsoidHeight); 


    std::cout << "Matrix" << std::endl;
    std::cout << std::fixed << returnMat << std::endl; 

    double distance = sqrt(pow(realPts[0] - C(0, 0), 2) + pow(realPts[1] - C(1, 0), 2) + pow(realPts[2] - C(2, 0), 2));
    std::cout << "Distance from Camera Center to Ground Truth Point" << std::endl;
    std::cout << std::fixed << distance << std::endl;

    double distance1 = sqrt(pow(C(0, 0) - returnMat(0, 0), 2) + pow(C(1, 0) - returnMat(1, 0), 2) + pow(C(2, 0) - returnMat(2, 0), 2));
    std::cout << "Distance from Camera Center to Estimated Point" << std::endl;
    std::cout << std::fixed << distance1 << std::endl;

    double distance2 = sqrt(pow(realPts[0] - returnMat(0, 0), 2) + pow(realPts[1] - returnMat(1, 0), 2) + pow(realPts[2] - returnMat(2, 0), 2));
    std::cout << "Error between Real Point and Estimated Point" << std::endl; 
    std::cout << std::fixed << distance2 << std::endl; 

    //Geocentric to Local Cartesian - Origin
    Eigen::Vector3d LCO_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(pts, latRad, lonRad, ellipsoidHeightC); 

    //Eigen::Vector3d realPts {{realPt(0, 0), realPt(1, 0), realPt(2, 0)}};
    Eigen::Vector3d estPts {{returnMat(0, 0), returnMat(1, 0), returnMat(2, 0)}};

    Eigen::Vector3d LCE_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(estPts, latRad, lonRad, ellipsoidHeightC); 
    Eigen::Vector3d LCM_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(realPts, latRad, lonRad, ellipsoidHeightC);



    //Angles
    // Angle between Realpoint and Estimated Point
    double angleRE = acos((LCM_3D_Pt[0] * LCE_3D_Pt(0) + LCM_3D_Pt[1] * LCE_3D_Pt(1) + LCM_3D_Pt[2] * LCE_3D_Pt(2)) / (
        (sqrt(pow(LCM_3D_Pt[0], 2) + pow(LCM_3D_Pt[1], 2) + pow(LCM_3D_Pt[2], 2))) *
        (sqrt(pow(LCE_3D_Pt(0), 2) + pow(LCE_3D_Pt(1), 2) + pow(LCE_3D_Pt(2), 2))))) * (180 / M_PI);

    // Angle between Estimated Point and Center
    double angleEC = acos((LCO_3D_Pt(0) * LCE_3D_Pt(0) + LCO_3D_Pt(1) * LCE_3D_Pt(1) + LCO_3D_Pt(2) * LCE_3D_Pt(2)) / (
        (sqrt(pow(LCO_3D_Pt(0), 2) + pow(LCO_3D_Pt(1), 2) + pow(LCO_3D_Pt(2), 2))) * 
        (sqrt(pow(LCE_3D_Pt(0), 2) + pow(LCE_3D_Pt(1), 2) + pow(LCE_3D_Pt(2), 2)))));

    // Angle between Realpoint and Center
    double angleRC = acos((LCM_3D_Pt[0] * LCO_3D_Pt(0) + LCM_3D_Pt[1] * LCO_3D_Pt(1) + LCM_3D_Pt[2] * LCO_3D_Pt(2)) / (
        (sqrt(pow(LCM_3D_Pt[0], 2) + pow(LCM_3D_Pt[1], 2) + pow(LCM_3D_Pt[2], 2))) * 
        (sqrt(pow(LCO_3D_Pt(0), 2) + pow(LCO_3D_Pt(1), 2) + pow(LCO_3D_Pt(2), 2)))));

    //std::cout << "Center Point " << std::endl;
    //std::cout << std::fixed << pts << std::endl; 

    std::cout << "Center Point LC" << std::endl;
    std::cout << std::fixed << LCO_3D_Pt << std::endl; 

    std::cout << "Estimated Point LC" << std::endl;
    std::cout << std::fixed << LCE_3D_Pt << std::endl;
    
    std::cout << "Real 3D Point LC" << std::endl;
    std::cout << std::fixed << LCM_3D_Pt << std::endl;

    std::cout << std::endl << "(" << LCM_3D_Pt[0] << "," << LCM_3D_Pt[1] << "," << LCM_3D_Pt[2] << ")" << " " 
        << "(" << LCE_3D_Pt[0] << "," << LCE_3D_Pt[1] << "," << LCE_3D_Pt[2] << ")" << " "
        << distance << " " << distance1 << " " << distance2 << " " << angleRE << " " << angleEC << " " << angleRC; 
}