#define _USE_MATH_DEFINES

#include "rotation.h"
#include "geospatial.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>

//Helper Functions
typedef Eigen::Matrix< long double, Eigen::Dynamic, Eigen::Dynamic > Mat;
typedef Eigen::Matrix< long double, Eigen::Dynamic, 1              > Vec; 

MatrixXd camera_pose(Eigen::MatrixXd Xcam, Eigen::MatrixXd X, Eigen::MatrixXd R) {
    Eigen::RowVectorXd X_x { {X(0, 0), X(0, 1), X(0, 2)} };
    Eigen::RowVectorXd XC_x { {Xcam(0, 0), Xcam(0, 1), Xcam(0, 2)} };
    Eigen::RowVectorXd X_y { {X(1, 0), X(1, 1), X(1, 2)} };
    Eigen::RowVectorXd XC_y { {Xcam(1, 0), Xcam(1, 1), Xcam(1, 2)} };
    Eigen::RowVectorXd X_z { {X(2, 0), X(2, 1), X(2, 2)} };
    Eigen::RowVectorXd XC_z { {Xcam(2, 0), Xcam(2, 1), Xcam(2, 2)} };
    Vector3d mu_XC{ {XC_x.mean(), XC_y.mean(), XC_z.mean()}};
    Vector3d mu_X{ {X_x.mean(), X_y.mean(), X_z.mean()} };

    //std::cout << XCam.col(0) - mu_XC << std::endl;
    std::cout << "Test Mu_X" << std::endl;
    std::cout << mu_X << std::endl; 
    std::cout << "Test X" << std::endl;
    std::cout << X << std::endl;
    std::cout << "Test X_x" << std::endl;
    std::cout << X_x.mean() << std::endl;
    std::cout << "Test R" << std::endl;
    std::cout << R << std::endl;
    std::cout << "Test Value" << std::endl;
    std::cout << (R * mu_X) << std::endl;
    //Vec t(3);
    //t = mu_XC - (R * mu_X);
    //std::cout << "Test Value Translation" << std::endl;
    //std::cout << t << std::endl;
    Vector3d translation = mu_XC - (R * mu_X);

    MatrixXd R_T(R.rows(), R.cols() + translation.cols());

    R_T << R, translation;

    return R_T;
}

MatrixXd dehomogenize(MatrixXd x) {
    MatrixXd x1Inhomog = x.block(0, 0, x.rows() - 1, x.cols());

    Eigen::RowVectorXd lastRow = x.block(x.rows() - 1, 0, 1, x.cols());
    //std::cout << "X" << std::endl;
    //std::cout << x << std::endl;
    //std::cout << "inhomogoenous" << std::endl;
    //std::cout << x1Inhomog << std::endl;
    //std::cout << "last Row" << std::endl;
    //std::cout << lastRow << std::endl;

    MatrixXd xinHomog = x1Inhomog.array().rowwise() / lastRow.array();

    return xinHomog;
}

/*
int main() {
    const char* dPath = geospatialDataDir();
    std::cout << dPath << std::endl;
    
    
    // First Camera Center
    double lon = -117.234208;
    double lat = 32.877139;


    // Original 3D Point
    //double lonPt = -117.234009; //+ 360.0;
    //double latPt = 32.876893; 

    // Test Points
    //// 30.04 Meters From first camera Center
    double lonPt = -117.234019;
    double latPt = 32.876920;

    //// 20 Meters From first camera Center
    //double lonPt = -117.234069;
    //double latPt = 32.877002;

    // 15.06 Meters From first camera Center
    //double lonPt = -117.234103; 
    //double latPt = 32.877036; 

    //// 10.02 Meters From first camera Center
    //double lonPt = -117.234129;
    //double latPt = 32.877078;

    // 5 Meters From first camera Center
    //double lonPt = -117.234162; 
    //double latPt = 32.877116; 

    // //3D Point moved up to 17 m
    //double lonPt = -117.234094;
    //double latPt = 32.877013; 

    //// 3D Point moved to 50m
    //double lonPt = -117.233856; 
    //double latPt = 32.876789;


    double lonRad = lon * (M_PI / 180);
    double latRad = lat * (M_PI / 180);

    double lonRadPt = lonPt * (M_PI / 180);
    double latRadPt = latPt * (M_PI / 180);


    //double latRad = -2.046122925563;
    std::cout << lonRad << std::endl;
    std::cout << latRad << std::endl;
    setGeospatialDataDir(dPath);


    double ellipsoidHeight = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRad, lonRad, 104);

    double ellipsoidHeightPt = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRadPt, lonRadPt, 104);

    std::cout << "Camera Height" << std::endl;
    std::cout << ellipsoidHeight << std::endl;

    std::cout << "3D Point Height" << std::endl;
    std::cout << ellipsoidHeightPt << std::endl;
    std::cout << std::endl;


    Eigen::Vector3d pts = WGS84GeodeticToWGS84Geocentric(latRad, lonRad, ellipsoidHeight);
    Eigen::Vector3d pts_3D = WGS84GeodeticToWGS84Geocentric(latRadPt, lonRadPt, ellipsoidHeightPt);


    // 1st Camera Setup
    std::cout << "Geocentric 1st Camera Origin Point" << std::endl;
    std::cout << std::fixed << pts << std::endl;

    std::cout << "Geocentric 3D Point" << std::endl;
    std::cout << std::fixed << pts_3D << std::endl;

    std::cout << std::endl;


    Eigen::Vector3d LCOpts = WGS84GeocentricToWGS84LocalCartesian(pts, latRad, lonRad, ellipsoidHeight);

    std::cout << "Local Cartesian Origin Point" << std::endl;
    std::cout << LCOpts << std::endl;
    std::cout << std::endl;


    Eigen::Vector3d xPt {1, 0, 0};
    Eigen::Vector3d yPt {0, 1, 0};
    Eigen::Vector3d zPt {0, 0, 1};

    MatrixXd C(xPt.rows(), LCOpts.cols() + xPt.cols() + yPt.cols() + zPt.cols());
    C << xPt, yPt, zPt, LCOpts;
    Eigen::MatrixXd pDst = MatrixXd::Zero(3, 4);


    std::cout << "Local Cartesian Points" << std::endl;
    std::cout << C << std::endl;
    std::cout << std::endl;

    WGS84LocalCartesianToWGS84Geocentric(C, pDst, C.cols(), latRad, lonRad, ellipsoidHeight);


    std::cout << "Geocentric Points of 1st Camera" << std::endl;
    std::cout << std::fixed << pDst << std::endl;
    std::cout << std::endl;


    // Second Camera Setup
    //double inch = 13 * 0.0254; //3(1) Inch Baseline
    //double inch = 0.095; //Realsense Baseline
    double inch = 0.20;


    Eigen::MatrixXd C2 = C; //inch + C.array();
    Eigen::RowVector4d incre {inch, inch, inch, inch};
    C2.row(0) = C2.row(0) + incre;

    std::cout << "2nd Camera Local Cartesian Points" << std::endl;
    std::cout << C2 << std::endl;
    std::cout << std::endl;

    Eigen::Vector3d C2Center = C2.col(3);
    std::cout << C2Center << std::endl;
    std::cout << std::endl;

    Eigen::MatrixXd pDst2 = MatrixXd::Zero(3, 4);

    WGS84LocalCartesianToWGS84Geocentric(C2, pDst2, C2.cols(), latRad, lonRad, ellipsoidHeight);

    std::cout << "2nd Camera Geocentric Points" << std::endl;
    std::cout << std::fixed << pDst2 << std::endl;

    std::cout << std::endl;

    Eigen::Vector3d LC_3DPt = WGS84GeocentricToWGS84LocalCartesian(pts_3D, latRad, lonRad, ellipsoidHeight);

    std::cout << "3D Local Cartesian Point" << std::endl;
    std::cout << std::fixed << LC_3DPt << std::endl;
    //std::cout << std::fixed << LC_3DPt[0] << std::endl;
    std::cout << std::endl;




    //std::cout << "3D Point to make light source" << std::endl;
    //double radiusForLightSource = 20.0 * 0.01; //3(1) Inch Baseline

    //Eigen::MatrixXd LC_3DCircle = LC_3DPt; //inch + C.array();
    ////Eigen::RowVectorXd increment {radiusForLightSource}; 
    //LC_3DCircle(0,0) = LC_3DCircle(0,0) + radiusForLightSource; 
    //std::cout << std::fixed << LC_3DCircle << std::endl;  
    //std::cout << std::endl;  

    //Eigen::Vector3d newPoint = WGS84LocalCartesianToWGS84Geocentric(LC_3DCircle, latRad, lonRad, ellipsoidHeight);
    //std::cout << "3D Point to make light source GeoCentric" << std::endl;
    //std::cout << std::fixed << newPoint << std::endl;
    //std::cout << std::endl; 




    double pitch = atan2(sqrt(pow(LC_3DPt[0], 2) + pow(LC_3DPt[0], 2)), LC_3DPt[2]);
    double yaw = atan2(LC_3DPt[1], LC_3DPt[0]);

    std::cout << "Yaw and Pitch of Camera 1" << std::endl;
    std::cout << std::fixed << "Yaw: " << yaw << " | Pitch: " << pitch << std::endl;
    //std::cout << std::fixed << LC_3DPt[0] << std::endl;
    std::cout << std::endl;

    double pitch2 = atan2(sqrt(pow(C2Center[0] - LC_3DPt[0], 2) + pow(C2Center[1] - LC_3DPt[0], 2)), LC_3DPt[2]);
    double yaw2 = atan2(C2Center[1] - LC_3DPt[1], C2Center[0] - LC_3DPt[0]);

    std::cout << "Yaw and Pitch of Camera 2" << std::endl;
    std::cout << std::fixed << "Yaw: " << yaw2 << " | Pitch: " << pitch2 << std::endl;
    //std::cout << std::fixed << LC_3DPt[0] << std::endl;
    std::cout << std::endl;

    double C2LatRad = 0.0;
    double C2LonRad = 0.0;
    double C2e_h = 0.0;

    WGS84LocalCartesianToWGS84Geodetic(C2Center,
        C2LatRad, C2LonRad, C2e_h, latRad, lonRad, ellipsoidHeight);


    std::cout << "Longitude, Latitude of Camera 2" << std::endl;
    std::cout << std::fixed << "Long: " << C2LonRad << " | Lat: " << C2LatRad << std::endl;
    std::cout << std::endl;

    double roll = 0.0;
    Eigen::Vector3d gimbal {0, M_PI / 2, 0}; //Yaw, Pitch, Roll

    // Params for Rotation.Geocentric to CameraFrame
    // Cam1: lonRad, latRad, gimbal[1], gimbal[2], gimbal[0], pitch, 0.0, yaw
    // Cam2: C2lonRad, C2latRad, gimbal[1], gimbal[2], gimbal[0], pitch2, 0.0, yaw2

    Rotation R = Rotation();
    MatrixXd R1 = R.rotationFromWGS84GeocentricToCameraFame(latRad, lonRad, gimbal[1], gimbal[2], gimbal[0], pitch, 0.0, yaw);
    std::cout << "Rotation of first Camera" << std::endl;
    std::cout << R1 << std::endl;


    MatrixXd R2 = R.rotationFromWGS84GeocentricToCameraFame(C2LatRad, C2LonRad, gimbal[1], gimbal[2], gimbal[0], pitch2, 0.0, yaw2);
    std::cout << "Rotation of second Camera" << std::endl;
    std::cout << R2 << std::endl;



    //// Get the rotation and translation of each camera projection matrix 


    //MatrixXd R_T1 = camera_pose(C, pDst, R1);

    //std::cout << "Rotation/Translation of first Camera" << std::endl;
    //std::cout << R_T1 << std::endl;


    //MatrixXd R_T2 = camera_pose(C2, pDst2, R2);

    //std::cout << "Rotation/Translation of second Camera" << std::endl;
    //std::cout << R_T2 << std::endl;

    //MatrixXd calibration{ {154.0966799187809, 0, 639.5}, {0, 154.0966799187809, 359.5}, {0, 0, 1} };


    //MatrixXd P1 = calibration * R_T1;
    //MatrixXd P2 = calibration * R_T2;


    //std::cout << "Camera Pose of First Camera" << std::endl;
    //std::cout << P1 << std::endl;

    //std::cout << "Camera Pose of Second Camera" << std::endl;
    //std::cout << P2 << std::endl;

    //VectorXd homog3DPt{ {pts_3D[0], pts_3D[1] , pts_3D[2], 1.0 } };
    //std::cout << std::endl << homog3DPt << std::endl;

    // Distance between Origin of First Camera to 3D point

    Eigen::Vector3d LCO_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(pts_3D, latRad, lonRad, ellipsoidHeight);

    double distance = sqrt(pow(LCO_3D_Pt[0] - LCOpts[0], 2) + pow(LCO_3D_Pt[1] - LCOpts[1], 2) + pow(LCO_3D_Pt[2] - LCOpts[2], 2));
    
    std::cout << "Distance of 3D point and First Camera" << std::endl;
    std::cout << distance << std::endl;

    //// Getting the Light Source on Camera

    //// Camera Rotation

    //std::ofstream myfile;
    //myfile.open("exampleLC33.txt", std::ofstream::app);

    //myfile << std::to_string(R1(0, 0)) << "," << std::to_string(R1(0, 1)) << "," << std::to_string(R1(0, 2)) << ",";
    //myfile << std::to_string(R1(1, 0)) << "," << std::to_string(R1(1, 1)) << "," << std::to_string(R1(1, 2)) << ","; 
    //myfile << std::to_string(R1(2, 0)) << "," << std::to_string(R1(2, 1)) << "," << std::to_string(R1(2, 2)) << "\n";

    //// Center Point  
    //myfile << std::to_string(pts_3D(0, 0)) << "," << std::to_string(pts_3D(1, 0)) << "," << std::to_string(pts_3D(2, 0)) << "\n";

    //// Radius Point
    //myfile << std::to_string(newPoint(0, 0)) << "," << std::to_string(newPoint(1, 0)) << "," << std::to_string(newPoint(2, 0)) << "\n";
    //myfile.close(); 


    //MatrixXd xPt1 = P1 * homog3DPt;
    //MatrixXd xPt2 = P2 * homog3DPt;
    //MatrixXd xPt1_inhomog = dehomogenize(xPt1); 
    //MatrixXd xPt2_inhomog = dehomogenize(xPt2);
    //std::cout << "2D Point from Camera Pose of First Camera" << std::endl;
    //std::cout << xPt1_inhomog << std::endl;
    //std::cout << std::endl; 

    //std::cout << "2D Point from Camera Pose of Second Camera" << std::endl;
    //std::cout << xPt2_inhomog << std::endl;
    //std::cout << std::endl; 

    //C, pDst, R1

    
    std::ofstream myfile;
    myfile.open("extrinsics30m.csv", std::ofstream::app); 

    myfile << std::to_string(C(0, 0)) << "," << std::to_string(C(0, 1)) << "," << std::to_string(C(0, 2)) << ","; 
    myfile << std::to_string(C(1, 0)) << "," << std::to_string(C(1, 1)) << "," << std::to_string(C(1, 2)) << ","; 
    myfile << std::to_string(C(2, 0)) << "," << std::to_string(C(2, 1)) << "," << std::to_string(C(2, 2)) << "\n"; 

    myfile << std::to_string(pDst(0, 0)) << "," << std::to_string(pDst(0, 1)) << "," << std::to_string(pDst(0, 2)) << ","; 
    myfile << std::to_string(pDst(1, 0)) << "," << std::to_string(pDst(1, 1)) << "," << std::to_string(pDst(1, 2)) << ","; 
    myfile << std::to_string(pDst(2, 0)) << "," << std::to_string(pDst(2, 1)) << "," << std::to_string(pDst(2, 2)) << "\n";

    myfile << std::to_string(R1(0, 0)) << "," << std::to_string(R1(0, 1)) << "," << std::to_string(R1(0, 2)) << ","; 
    myfile << std::to_string(R1(1, 0)) << "," << std::to_string(R1(1, 1)) << "," << std::to_string(R1(1, 2)) << ","; 
    myfile << std::to_string(R1(2, 0)) << "," << std::to_string(R1(2, 1)) << "," << std::to_string(R1(2, 2)) << "\n"; 

    myfile << std::to_string(C2(0, 0)) << "," << std::to_string(C2(0, 1)) << "," << std::to_string(C2(0, 2)) << ",";
    myfile << std::to_string(C2(1, 0)) << "," << std::to_string(C2(1, 1)) << "," << std::to_string(C2(1, 2)) << ",";
    myfile << std::to_string(C2(2, 0)) << "," << std::to_string(C2(2, 1)) << "," << std::to_string(C2(2, 2)) << "\n";

    myfile << std::to_string(pDst2(0, 0)) << "," << std::to_string(pDst2(0, 1)) << "," << std::to_string(pDst2(0, 2)) << ",";
    myfile << std::to_string(pDst2(1, 0)) << "," << std::to_string(pDst2(1, 1)) << "," << std::to_string(pDst2(1, 2)) << ",";
    myfile << std::to_string(pDst2(2, 0)) << "," << std::to_string(pDst2(2, 1)) << "," << std::to_string(pDst2(2, 2)) << "\n";

    myfile << std::to_string(R2(0, 0)) << "," << std::to_string(R2(0, 1)) << "," << std::to_string(R2(0, 2)) << ",";
    myfile << std::to_string(R2(1, 0)) << "," << std::to_string(R2(1, 1)) << "," << std::to_string(R2(1, 2)) << ",";
    myfile << std::to_string(R2(2, 0)) << "," << std::to_string(R2(2, 1)) << "," << std::to_string(R2(2, 2)) << "\n";

    //myfile << std::to_string(P1(0, 0)) << "," << std::to_string(P1(0, 1)) << "," << std::to_string(P1(0, 2)) << "," << std::to_string(P1(0, 3)) << ",";
    //myfile << std::to_string(P1(1, 0)) << "," << std::to_string(P1(1, 1)) << "," << std::to_string(P1(1, 2)) << "," << std::to_string(P1(1, 3)) << ",";
    //myfile << std::to_string(P1(2, 0)) << "," << std::to_string(P1(2, 1)) << "," << std::to_string(P1(2, 2)) << "," << std::to_string(P1(2, 3)) << "\n";

    //myfile << std::to_string(P2(0, 0)) << "," << std::to_string(P2(0, 1)) << "," << std::to_string(P2(0, 2)) << "," << std::to_string(P2(0, 3)) << ",";
    //myfile << std::to_string(P2(1, 0)) << "," << std::to_string(P2(1, 1)) << "," << std::to_string(P2(1, 2)) << "," << std::to_string(P2(1, 3)) << ",";
    //myfile << std::to_string(P2(2, 0)) << "," << std::to_string(P2(2, 1)) << "," << std::to_string(P2(2, 2)) << "," << std::to_string(P2(2, 3 )) << "\n";

    ////myfile << std::to_string(xPt1_inhomog(0, 0)) << "," << std::to_string(xPt1_inhomog(1, 0)) << "\n";
    ////myfile << std::to_string(xPt2_inhomog(0, 0)) << "," << std::to_string(xPt2_inhomog(1, 0)) << "\n";
    myfile.close();

    // Saving Geocentric Points
    //5m first - 30m point

    //std::ofstream myfile; 
    //myfile.open("geocentricPts.csv", std::ofstream::app);

    //myfile << std::to_string(pts_3D(0, 0)) << "," << std::to_string(pts_3D(1, 0)) << "," << std::to_string(pts_3D(2, 0)) << "\n"; 
    //myfile.close(); 
}*/


//Test gps to geocentric

/*
int main() {
    const char* dPath = geospatialDataDir();
    std::cout << dPath << std::endl;

    double lon = -117.234517;
    double lat = 32.882108;

    // UCSD Experiment
    // Original 3D Point 5m Point
    //double lonPt = -117.234553;
    //double latPt = 32.882141; 

    // Original 3D Point 10m Point
    //double lonPt = -117.234613;
    //double latPt = 32.882148;

    // Original 3D Point 15m Point
    double lonPt = -117.234622;
    double latPt = 32.882201;

    // //3D Point moved up to 17 m
    //double lonPt = -117.234094;
    //double latPt = 32.877013; 

    //// 3D Point moved to 50m
    //double lonPt = -117.233856; 
    //double latPt = 32.876789;


    double lonRad = lon * (M_PI / 180);
    double latRad = lat * (M_PI / 180);

    double lonRadPt = lonPt * (M_PI / 180);
    double latRadPt = latPt * (M_PI / 180);


    //double latRad = -2.046122925563;
    std::cout << lonRad << std::endl;
    std::cout << latRad << std::endl;
    setGeospatialDataDir(dPath);


    double ellipsoidHeight = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRad, lonRad, 108);

    double ellipsoidHeightPt = orthometricHeightToWGS84EllipsoidHeight_EGM2008(latRadPt, lonRadPt, 108);

    std::cout << "Camera Height" << std::endl;
    std::cout << ellipsoidHeight << std::endl;

    std::cout << "3D Point Height" << std::endl;
    std::cout << ellipsoidHeightPt << std::endl;
    std::cout << std::endl;


    Eigen::Vector3d pts = WGS84GeodeticToWGS84Geocentric(latRad, lonRad, ellipsoidHeight);
    Eigen::Vector3d pts_3D = WGS84GeodeticToWGS84Geocentric(latRadPt, lonRadPt, ellipsoidHeightPt);


    // 1st Camera Setup
    std::cout << "Geocentric 1st Camera Origin Point" << std::endl;
    std::cout << std::fixed << pts << std::endl;

    std::cout << "Geocentric 3D Point" << std::endl;
    std::cout << std::fixed << pts_3D << std::endl;

    //Geocentric to Local Cartesian
    Eigen::Vector3d LCO_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(pts, latRad, lonRad, ellipsoidHeight);

    //Eigen::Vector3d est_pt {   { -2434442.290780 },
    //                    { -4747150.816758 },
    //                    { 3483887.031847 }};

    Eigen::Vector3d LC_3D_Pt = WGS84GeocentricToWGS84LocalCartesian(pts_3D, latRad, lonRad, ellipsoidHeight); 

    double distance = sqrt(pow(LCO_3D_Pt[0] - LC_3D_Pt[0], 2) + pow(LCO_3D_Pt[1] - LC_3D_Pt[1], 2) + pow(LCO_3D_Pt[2] - LC_3D_Pt[2], 2)); 
    std::cout << "Distance" << std::endl;
    std::cout << std::fixed << distance << std::endl; 

    double distance2 = sqrt(pow(pts[0] - pts_3D[0], 2) + pow(pts[1] - pts_3D[1], 2) + pow(pts[2] - pts_3D[2], 2));
    std::cout << "Distance" << std::endl; 
    std::cout << std::fixed << distance2 << std::endl;
}*/