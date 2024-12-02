#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Matrix3d; 
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::RowVector4d;

class TPGeo {
public:
	//open_image(string imagePath);
	double sinc(double x);
	Matrix3d skew(Vector3d w);
	Matrix3d deparameterize_Omega(Vector3d w);
	void parameterize_Rotation(Matrix3d r, Vector3d *w, double *theta); // CHANGE TO INCLUDE REFERENCES AND FILL REFERENCES 
	MatrixXd calc_camera_proj_matrix(Matrix3d K, Matrix3d R, VectorXd t);
	MatrixXd homogenize(MatrixXd x);
	MatrixXd dehomogenize(MatrixXd x);
	Matrix3d angle_axis_to_rotation(double angle, MatrixXd axis);
	Matrix3d interpolate_rotation(double alpha, Matrix3d r1, Matrix3d r2);
	Vector3d camera_translation(Matrix3d R1, VectorXd t1, Matrix3d R2);
	Matrix3d calc_projective_transformation(Matrix3d K1, Matrix3d R1, Matrix3d K2, Matrix3d R2);
	void epipolar_rectification(Matrix3d K1, Matrix3d R1, Vector3d t1, Matrix3d K2, Matrix3d R2, Vector3d t2,
		Matrix3d *Krectified, Matrix3d *Rrectified, Vector3d *t1rectified, Vector3d *t2rectified, Matrix3d *H1, Matrix3d *H2);
	MatrixXd calc_epipole(MatrixXd E);
	void backprojection(VectorXd x, MatrixXd P, RowVector4d *row1, RowVector4d *row2);
	MatrixXd triangulation2View(VectorXd x1, VectorXd x2, MatrixXd P1rectified, MatrixXd P2rectified);
	//geocentric_triangulation2View(left_img, right_img, x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
	//rectified_calibration_matrices(Krectified, H1, H2, I1, I2)
};