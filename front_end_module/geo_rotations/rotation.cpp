#include "rotation.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::Matrix3d;

Eigen::Matrix3d Rotation::eulerAnglesZXYToRotationMatrix(double gamma, double alpha, double beta){
	Eigen::Matrix3d R;
	double sin_gamma = sin(gamma);
	double cos_gamma = cos(gamma);
	double sin_alpha = sin(alpha);
	double cos_alpha = cos(alpha);
	double sin_beta = sin(beta);
	double cos_beta = cos(beta);

	R(0, 0) = cos_beta * cos_gamma - sin_alpha * sin_beta * sin_gamma;
	R(0, 1) = -cos_alpha * sin_gamma;
	R(0, 2) = cos_gamma * sin_beta + cos_beta * sin_alpha * sin_gamma;
	R(1, 0) = cos_beta * sin_gamma + cos_gamma * sin_alpha * sin_beta;
	R(1, 1) = cos_alpha * cos_gamma;
	R(1, 2) = sin_beta * sin_gamma - cos_beta * cos_gamma * sin_alpha;
	R(2, 0) = -cos_alpha * sin_beta;
	R(2, 1) = sin_alpha;
	R(2, 2) = cos_alpha * cos_beta;

	return R;
}

Matrix3d Rotation::eulerAnglesZYXToRotationMatrix(double gamma, double beta, double alpha){
	Eigen::Matrix3d R = Matrix3d::Zero();
	double sin_gamma = sin(gamma);
	double cos_gamma = cos(gamma);
	double sin_alpha = sin(alpha);
	double cos_alpha = cos(alpha);
	double sin_beta = sin(beta);
	double cos_beta = cos(beta);

	R(0, 0) = cos_beta * cos_gamma;
	R(0, 1) = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma;
	R(0, 2) = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma;
	R(1, 0) = cos_beta * sin_gamma;
	R(1, 1) = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma;
	R(1, 2) = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma;
	R(2, 0) = -sin_beta;
	R(2, 1) = cos_beta * sin_alpha;
	R(2, 2) = cos_beta * cos_alpha;
	/*Eigen::Matrix3d R{  {cos_beta * cos_gamma, sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma, cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma},
						{cos_beta * sin_gamma, sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma, cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma},
						{-sin_beta, cos_beta * sin_alpha, cos_beta * cos_alpha} };*/


	std::cout << "ZYX" << std::endl;
	std::cout << R << std::endl;
	std::cout << sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma << std::endl;
	std::cout << cos_beta * cos_alpha << std::endl;


	return R;
}
Matrix3d Rotation::rotationFromWGS84GeocentricToWGS84LocalCartesian(double originLatitude, double originLongitude) {
	Eigen::Matrix3d R;
	double sin_lat = sin(originLatitude);
	double cos_lat = cos(originLatitude);
	double sin_lon = sin(originLongitude);
	double cos_lon = cos(originLongitude);

	R(0, 0) = -sin_lon;
	R(0, 1) = cos_lon;
	R(0, 2) = 0;
	R(1, 0) = -sin_lat * cos_lon;
	R(1, 1) = -sin_lat * sin_lon;
	R(1, 2) = cos_lat;
	R(2, 0) = cos_lat * cos_lon;
	R(2, 1) = cos_lat * sin_lon;
	R(2, 2) = sin_lat;

	return R;
}
Matrix3d Rotation::rotX(double theta) {
	Eigen::Matrix3d R = Matrix3d::Identity();

	R(1, 1) = cos(theta);
	R(1, 2) = -sin(theta);
	R(2, 1) = sin(theta);
	R(2, 2) = cos(theta);
	std::cout << "X" << std::endl;
	std::cout << R << std::endl;

	return R;
}
Matrix3d Rotation::rotY(double theta) {
	Eigen::Matrix3d R = Matrix3d::Identity();

	R(0, 0) = cos(theta);
	R(0, 2) = sin(theta);
	R(2, 0) = -sin(theta);
	R(2, 2) = cos(theta);
	std::cout << "Y" << std::endl;
	std::cout << R << std::endl;
	return R;
}
Matrix3d Rotation::rotZ(double theta) {
	Eigen::Matrix3d R = Matrix3d::Identity();

	R(0, 0) = cos(theta);
	R(0, 1) = -sin(theta);
	R(1, 0) = sin(theta);
	R(1, 1) = cos(theta);
	std::cout << "Z" << std::endl;
	std::cout << R << std::endl;
	return R;
}
Matrix3d Rotation::rotationFromWGS84GeocentricToCameraFame(double latitude_radians, double longitude_radians, double gimbal_pitch, double gimbal_roll, double gimbal_yaw, double platform_pitch, double platform_roll, double platform_yaw) {
	Eigen::Matrix3d Rba { {0, 1, 0}, {0, 0, 1}, {1, 0 ,0 } };

	Eigen::Matrix3d Rcb = eulerAnglesZYXToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll);
	// iPhone version
	//Eigen::Matrix3d Rcb = eulerAnglesZXYToRotationMatrix(gimbal_yaw, gimbal_pitch, gimbal_roll); 


	Rcb.transposeInPlace(); 

	std::cout << "RCB" << std::endl;
	std::cout << Rcb << std::endl;
	Eigen::Matrix3d Rca = Rba * Rcb;
	
	Eigen::Matrix3d Rdc{{0, 1, 0}, {1, 0, 0}, {0, 0 ,-1 }};

	Eigen::Matrix3d Rda = Rca * Rdc;

	Eigen::Matrix3d Red = rotY(platform_roll) * rotX(platform_pitch) * rotZ(platform_yaw);
	std::cout << "RED" << std::endl;
	std::cout << Red << std::endl;
	Eigen::Matrix3d Rea = Rda * Red;

	Matrix3d Rfe = Matrix3d::Identity();
	Matrix3d Rfa = Rea * Rfe;

	std::cout << "RFA" << std::endl;
	std::cout << Rfa << std::endl;

	Matrix3d Rgf = rotationFromWGS84GeocentricToWGS84LocalCartesian(latitude_radians, longitude_radians);
	std::cout << "RGF" << std::endl;
	std::cout << Rgf << std::endl;

	std::cout << std::endl;
	Matrix3d Rga = Rfa * Rgf;

	return Rga;
}

//int main()
//{
//	Rotation R = Rotation();
//	MatrixXd m = R.rotationFromWGS84GeocentricToCameraFame(-2.0458482456501974, 0.5737776897248366, -1.5707963267948966, 0, 0, 0, 0, 0); //Matches
//	//MatrixXd m = R.rotationFromWGS84GeocentricToCameraFame(-2.0458482456501974, 0.5737776897248366, -1.5707963267948966, 0, 0, -0.1, 0.1, 3);
//	std::cout << m << std::endl;
//}