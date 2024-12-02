#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;

class Rotation {
public:
	Matrix3d eulerAnglesZXYToRotationMatrix(double gamma, double  alpha, double beta);
	Matrix3d eulerAnglesZYXToRotationMatrix(double gamma, double  beta, double alpha);
	Matrix3d rotationFromWGS84GeocentricToWGS84LocalCartesian(double originLatitude, double originLongitude);
	Matrix3d rotX(double theta);
	Matrix3d rotY(double theta);
	Matrix3d rotZ(double theta);
	Matrix3d rotationFromWGS84GeocentricToCameraFame(double latitude_radians, double longitude_radians, double gimbal_pitch, double gimbal_roll, double gimbal_yaw, double platform_pitch, double platform_roll, double platform_yaw);
	//double dot_product(Eigen::VectorXd v1, Eigen::VectorXd v2);
	//double length(Eigen::VectorXd v);
	//double angle(Eigen::VectorXd] v1, Eigen::VectorXd v2);
};