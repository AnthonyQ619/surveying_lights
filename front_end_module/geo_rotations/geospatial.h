////////////////////////////////////////////////////////////////////////////////
//                 Copyright (c) 2006-2021 Benjamin L. Ochoa
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Geospatial functions.
////////////////////////////////////////////////////////////////////////////////

#if !defined( _BLO_GEOSPATIAL_H_ )
#define _BLO_GEOSPATIAL_H_
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::RowVector3d;
using Eigen::Vector3d;
using Eigen::Vector4d;

//#include "defs.h"

//namespace blo
//{
template<typename T, size_t N> class Point;
template<typename T, size_t M, size_t N> class SmallMatrix;

////////////////////////////////////////////////////////////////////////////////
/// \brief   Gets the string containing the path to the geospatial data 
///          directory.
///
/// \details This function returns the string containing the path to the 
///          geospatial data directory.  This directory should contain the 
///          datum parameter files, ellipsoid parameters file, and geoid 
///          undulation data files.
///
/// \return  The string containing the path to the geospatial data directory.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
const char* geospatialDataDir();

////////////////////////////////////////////////////////////////////////////////
/// \brief   Sets the path to the geospatial data directory.
///
/// \details This function sets path to the geospatial data directory.  This 
///          directory should contain the datum parameter files, ellipsoid 
///          parameters file, and geoid undulation data files.
///
/// \param   dataDir The string containing the path to the geospatial data 
///                  directory.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
void setGeospatialDataDir( const char* dataDir );


////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 ellipsoid height to orthometric height using EGM2008.
///
/// \details This function converts WGS84 ellipsoid height at a specified 
///          latitude and longitude to orthometric height using EGM2008.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   latitude        The source WGS84 geodetic latitude in radians.
/// \param   longitude       The source WGS84 geodetic longitude in radians.
/// \param   ellipsoidHeight The source WGS84 ellipsoid height in meters.
///
/// \return  The orthometric height in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
double WGS84EllipsoidHeightToOrthometricHeight_EGM2008( 
    double latitude, double longitude, double ellipsoidHeight );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 ellipsoid height to orthometric height using EGM2008.
///
/// \details This function converts WGS84 ellipsoid height at a specified 
///          latitude and longitude to orthometric height using EGM2008.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   pSrcLatitude        Pointer to the source WGS84 geodetic latitudes 
///                              in radians.
/// \param   pSrcLongitude       Pointer to the source WGS84 geodetic 
///                              longitudes in radians.
/// \param   pSrcEllipsoidHeight Pointer to the source WGS84 ellipsoid heights 
///                              in meters.
/// \param   pDst                Pointer to the destination orthometric heights 
///                              in meters.
/// \param   count               Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
void WGS84EllipsoidHeightToOrthometricHeight_EGM2008( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcEllipsoidHeight, double* pDst, int count );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert orthometric height to WGS84 ellipsoid height using EGM2008.
///
/// \details This function converts orthometric height at a specified 
///          latitude and longitude to WGS84 ellipsoid height using EGM2008.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   latitude          The source WGS84 geodetic latitude in radians.
/// \param   longitude         The source WGS84 geodetic longitude in radians.
/// \param   orthometricHeight The source orthometric height in meters.
///
/// \return  The WGS84 ellipsoid height in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
double orthometricHeightToWGS84EllipsoidHeight_EGM2008( 
    double latitude, double longitude, double orthometricHeight );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert orthometric height to WGS84 ellipsoid height using EGM2008.
///
/// \details This function converts orthometric height at a specified latitude 
///          and longitude to WGS84 ellipsoid height using EGM2008.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   pSrcLatitude          Pointer to the source WGS84 geodetic 
///                                latitudes in radians.
/// \param   pSrcLongitude         Pointer to the source WGS84 geodetic 
///                                longitudes in radians.
/// \param   pSrcOrthometricHeight Pointer to the source orthometric heights in 
///                                meters.
/// \param   pDst                  Pointer to the destination WGS84 ellipsoid 
///                                heights in meters.
/// \param   count                 Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
void orthometricHeightToWGS84EllipsoidHeight_EGM2008( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcOrthometricHeight, double* pDst, int count );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 ellipsoid height to orthometric height using EGM96.
///
/// \details This function converts WGS84 ellipsoid height at a specified 
///          latitude and longitude to orthometric height using EGM96.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   latitude        The source WGS84 geodetic latitude in radians.
/// \param   longitude       The source WGS84 geodetic longitude in radians.
/// \param   ellipsoidHeight The source WGS84 ellipsoid height in meters.
///
/// \return  The orthometric height in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
double WGS84EllipsoidHeightToOrthometricHeight_EGM96( double latitude, 
    double longitude, double ellipsoidHeight );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 ellipsoid height to orthometric height using EGM96.
///
/// \details This function converts WGS84 ellipsoid height at a specified 
///          latitude and longitude to orthometric height using EGM96.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   pSrcLatitude        Pointer to the source WGS84 geodetic latitudes 
///                              in radians.
/// \param   pSrcLongitude       Pointer to the source WGS84 geodetic 
///                              longitudes in radians.
/// \param   pSrcEllipsoidHeight Pointer to the source WGS84 ellipsoid heights 
///                              in meters.
/// \param   pDst                Pointer to the destination orthometric heights 
///                              in meters.
/// \param   count               Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
void WGS84EllipsoidHeightToOrthometricHeight_EGM96( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcEllipsoidHeight, double* pDst, int count );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert orthometric height to WGS84 ellipsoid height using EGM96.
///
/// \details This function converts orthometric height at a specified latitude 
///          and longitude to WGS84 ellipsoid height using EGM96.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   latitude          The source WGS84 geodetic latitude in radians.
/// \param   longitude         The source WGS84 geodetic longitude in radians.
/// \param   orthometricHeight The source orthometric height in meters.
///
/// \return  The WGS84 ellipsoid height in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
double orthometricHeightToWGS84EllipsoidHeight_EGM96( double latitude, 
    double longitude, double orthometricHeight );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert orthometric height to WGS84 ellipsoid height using EGM96.
///
/// \details This function converts orthometric height at a specified latitude 
///          and longitude to WGS84 ellipsoid height using EGM96.
///
/// \note    Prior to the first call to any of the orthometric height 
///          conversion functions, the path to the geospatial data directory 
///          must be set to the directory containing the geoid undulation data 
///          files.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.  The distance from the 
///          point to the geoid is called the orthometric height.
///
/// \param   pSrcLatitude          Pointer to the source WGS84 geodetic 
///                                latitudes in radians.
/// \param   pSrcLongitude         Pointer to the source WGS84 geodetic 
///                                longitudes in radians.
/// \param   pSrcOrthometricHeight Pointer to the source orthometric heights in 
///                                meters.
/// \param   pDst                  Pointer to the destination WGS84 ellipsoid 
///                                heights in meters.
/// \param   count                 Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
void orthometricHeightToWGS84EllipsoidHeight_EGM96( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcOrthometricHeight, double* pDst, int count );

////////////////////////////////////////////////////////////////////////////////
/// \brief   Quadric representing the WGS84 reference ellipsoid.
///
/// \details This function returns the quadric in geocentric coordinates in 
///          meters that represents the World Geodetic System 1984 (WGS84) 
///          reference ellipsoid.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \return  The quadric.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
const Eigen::Matrix4d WGS84GeocentricQuadric();

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geodetic coordinates to WGS84 geocentric coordinates.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \param   latitude        The source WGS84 geodetic latitude in radians.
/// \param   longitude       The source WGS84 geodetic longitude in radians.
/// \param   ellipsoidHeight The source WGS84 ellipsoid height in meters.
///
/// \return  The destination point in WGS84 geocentric coordinates in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API const Point<double, 3> WGS84GeodeticToWGS84Geocentric( double latitude, 
//    double longitude, double ellipsoidHeight );

const Eigen::Vector3d WGS84GeodeticToWGS84Geocentric(double latitude,
    double longitude, double ellipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geodetic coordinates to WGS84 geocentric coordinates.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \param   pSrcLatitude        Pointer to the source WGS84 geodetic latitudes 
///                              in radians.
/// \param   pSrcLongitude       Pointer to the source WGS84 geodetic 
///                              longitudes in radians.
/// \param   pSrcEllipsoidHeight Pointer to the source WGS84 ellipsoid heights 
///                              in meters.
/// \param   pDst                Pointer to the destination points in WGS84 
///                              geocentric coordinates in meters.
/// \param   count               Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84GeodeticToWGS84Geocentric( const double* pSrcLatitude, 
//    const double* pSrcLongitude, const double* pSrcEllipsoidHeight, 
//    Point<double, 3>* pDst, int count );

void WGS84GeodeticToWGS84Geocentric(const double* pSrcLatitude,
    const double* pSrcLongitude, const double* pSrcEllipsoidHeight,
    Eigen::MatrixXd& pDst, int count);


////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geocentric coordinates to WGS84 geodetic coordinates.
///
/// \details This function converts WGS84 geocentric coordinates to WGS84 
///          geodetic coordinates using an iterative algorithm.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \param   src             The source point in WGS84 geocentric coordinates 
///                          in meters.
/// \param   latitude        The destination WGS84 geodetic latitude in radians.
/// \param   longitude       The destination WGS84 geodetic longitude in 
///                          radians.
/// \param   ellipsoidHeight The destination WGS84 ellipsoid height in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84GeocentricToWGS84Geodetic( const Point<double, 3>& src, 
//    double& latitude, double& longitude, double& ellipsoidHeight );

void WGS84GeocentricToWGS84Geodetic(const Eigen::Vector3d& src,
    double& latitude, double& longitude, double& ellipsoidHeight);


////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geocentric coordinates to WGS84 geodetic coordinates.
///
/// \details This function converts WGS84 geocentric coordinates to WGS84 
///          geodetic coordinates using an iterative algorithm.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \param   pSrc                Pointer to the source points in WGS84 
///                              geocentric coordinates in meters.
/// \param   pDstLatitude        Pointer to the destination WGS84 geodetic 
///                              latitudes in radians.
/// \param   pDstLongitude       Pointer to the destination WGS84 geodetic 
///                              longitudes in radians.
/// \param   pDstEllipsoidHeight Pointer to the destination WGS84 ellipsoid 
///                              heights in meters.
/// \param   count               Number of points.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84GeocentricToWGS84Geodetic( const Point<double, 3>* pSrc, 
//    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight, 
//    int count );

void WGS84GeocentricToWGS84Geodetic(const Eigen::MatrixXd& pSrc,
    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight,
    int count);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geodetic coordinates to WGS84 local Cartesian 
///          coordinates.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \param   latitude              The source WGS84 geodetic latitude in 
///                                radians.
/// \param   longitude             The source WGS84 geodetic longitude in 
///                                radians.
/// \param   ellipsoidHeight       The source WGS84 ellipsoid height in meters.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
///
/// \return  The destination point in WGS84 local Cartesian coordinates in 
///          meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API const Point<double, 3> WGS84GeodeticToWGS84LocalCartesian( 
//    double latitude, double longitude, double ellipsoidHeight, 
//    double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

const Eigen::Vector3d WGS84GeodeticToWGS84LocalCartesian(
    double latitude, double longitude, double ellipsoidHeight,
    double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geodetic coordinates to WGS84 local Cartesian 
///          coordinates.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \param   pSrcLatitude          Pointer to the source WGS84 geodetic 
///                                latitudes in radians.
/// \param   pSrcLongitude         Pointer to the source WGS84 geodetic 
///                                longitudes in radians.
/// \param   pSrcEllipsoidHeight   Pointer to the source WGS84 ellipsoid 
///                                heights in meters.
/// \param   pDst                  Pointer to the destination point in WGS84 
///                                local Cartesian coordinates in meters.
/// \param   count                 Number of points.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84GeodeticToWGS84LocalCartesian( const double* pSrcLatitude, 
//    const double* pSrcLongitude, const double* pSrcEllipsoidHeight, 
//    Point<double, 3>* pDst, int count, double originLatitude, 
//    double originLongitude, double originEllipsoidHeight );

void WGS84GeodeticToWGS84LocalCartesian(const double* pSrcLatitude,
    const double* pSrcLongitude, const double* pSrcEllipsoidHeight,
    Eigen::MatrixXd& pDst, int count, double originLatitude,
    double originLongitude, double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 local Cartesian coordinates to WGS84 geodetic 
///          coordinates.
///
/// \details This function converts WGS84 local Cartesian coordinates to WGS84 
///          geodetic coordinates using an iterative algorithm.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \param   src                   The source point in WGS84 local Cartesian 
///                                coordinates in meters.
/// \param   latitude              The destination WGS84 geodetic latitude in 
///                                radians.
/// \param   longitude             The destination WGS84 geodetic longitude in 
///                                radians.
/// \param   ellipsoidHeight       The destination WGS84 ellipsoid height in 
///                                meters.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84LocalCartesianToWGS84Geodetic( const Point<double, 3>& src, 
//    double& latitude, double& longitude, double& ellipsoidHeight, 
//    double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

void WGS84LocalCartesianToWGS84Geodetic(const Eigen::Vector3d& src,
    double& latitude, double& longitude, double& ellipsoidHeight,
    double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 local Cartesian coordinates to WGS84 geodetic 
///          coordinates.
///
/// \details This function converts WGS84 local Cartesian coordinates to WGS84 
///          geodetic coordinates using an iterative algorithm.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \note    WGS84 geodetic coordinates consist of the geodetic latitude and 
///          longitude, and ellipsoid height that define the position of a 
///          point on, or near, the surface of the earth with respect to the 
///          WGS84 reference ellipsoid.  Geodetic longitude is not defined when 
///          the point lies on the Z axis.  The distance from the point to the 
///          ellipsoid is called the ellipsoid height.
///
/// \param   pSrc                  Pointer to the source points in WGS84 local 
///                                Cartesian coordinates in meters.
/// \param   pDstLatitude          Pointer to the destination WGS84 geodetic 
///                                latitudes in radians.
/// \param   pDstLongitude         Pointer to the destination WGS84 geodetic 
///                                longitudes in radians.
/// \param   pDstEllipsoidHeight   Pointer to the destination WGS84 ellipsoid 
///                                heights in meters.
/// \param   count                 Number of points.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84LocalCartesianToWGS84Geodetic( const Point<double, 3>* pSrc, 
//    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight, 
//    int count, double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

void WGS84LocalCartesianToWGS84Geodetic(const Eigen::MatrixXd& pSrc,
    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight,
    int count, double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geocentric coordinates to WGS84 local Cartesian 
///          coordinates.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \param   src                   The source point in WGS84 geocentric 
///                                coordinates in meters.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
///
/// \return  The destination point in WGS84 local Cartesian coordinates in 
///          meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API const Point<double, 3> WGS84GeocentricToWGS84LocalCartesian( 
//    const Point<double, 3>& src, double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

const Eigen::Vector3d WGS84GeocentricToWGS84LocalCartesian(
    const Eigen::Vector3d& src, double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 geocentric coordinates to WGS84 local Cartesian 
///          coordinates.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \param   pSrc                  Pointer to the source points in WGS84 
///                                geocentric coordinates in meters.
/// \param   pDst                  Pointer to the destination points in WGS84 
///                                local Cartesian coordinates in meters.
/// \param   count                 Number of points.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API void WGS84GeocentricToWGS84LocalCartesian( 
//    const Point<double, 3>* pSrc, Point<double, 3>* pDst, int count, 
//    double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

void WGS84GeocentricToWGS84LocalCartesian(
    const Eigen::MatrixXd& pSrc, Eigen::MatrixXd& pDst, int count,
    double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Calculate the rotation from the WGS84 geocentric coordinate frame 
///          to the WGS84 local Cartesian coordinate frame.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \param   originLatitude  WGS84 geodetic latitude of the local origin in 
///                          radians.
/// \param   originLongitude WGS84 geodetic longitude of the local origin in 
///                          radians.
////////////////////////////////////////////////////////////////////////////////
//BLO_API const SmallMatrix<double, 3, 3> 
//    rotationFromWGS84GeocentricToWGS84LocalCartesian( double originLatitude, 
//    double originLongitude );
// IN ROTATION FILE ^^^

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 local Cartesian coordinates to WGS84 geocentric 
///          coordinates.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \param   src                   The source point in WGS84 local Cartesian 
///                                coordinates in meters.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
///
/// \return  The destination point in WGS84 geocentric coordinates in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API const Point<double, 3> WGS84LocalCartesianToWGS84Geocentric( 
//    const Point<double, 3>& src, double originLatitude, double originLongitude, 
//    double originEllipsoidHeight );

const Eigen::Vector3d WGS84LocalCartesianToWGS84Geocentric(
    const Eigen::Vector3d& src, double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Convert WGS84 local Cartesian coordinates to WGS84 geocentric 
///          coordinates.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \param   pSrc                  Pointer to the source points in WGS84 local 
///                                Cartesian coordinates in meters.
/// \param   pDst                  Pointer to the destination points in WGS84 
///                                geocentric coordinates in meters.
/// \param   count                 Number of points.
/// \param   originLatitude        WGS84 geodetic latitude of the local origin 
///                                in radians.
/// \param   originLongitude       WGS84 geodetic longitude of the local origin 
///                                in radians.
/// \param   originEllipsoidHeight WGS84 ellipsoid height of the local origin 
///                                in meters.
////////////////////////////////////////////////////////////////////////////////
//BLO_API 
 //void WGS84LocalCartesianToWGS84Geocentric( 
 //   const Point<double, 3>* pSrc, Point<double, 3>* pDst, int count, 
 //   double originLatitude, double originLongitude, 
 //   double originEllipsoidHeight );

void WGS84LocalCartesianToWGS84Geocentric(
    const Eigen::MatrixXd& pSrc, Eigen::MatrixXd& pDst, int count,
    double originLatitude, double originLongitude,
    double originEllipsoidHeight);

////////////////////////////////////////////////////////////////////////////////
/// \brief   Calculate the rotation from the WGS84 local Cartesian coordinate 
///          frame to the WGS84 geocentric coordinate frame.
///
/// \note    The WGS84 local Cartesian coordinate system is defined by the 
///          location of a specified origin in geodetic coordinates.  If the 
///          specified origin ellipsoid height is zero, the local XY plane is 
///          tangent to the surface of the WGS84 reference ellipsoid at the 
///          specified latitude and longitude.  If the origin height is 
///          non-zero, then the local XY plane is shifted (up or down) 
///          accordingly.  The local positive X axis points east, the local 
///          positive Y axis points north, and the local positive Z axis points 
///          up, orthogonal to the ellipsoid surface at the specified origin.
///
/// \note    WGS84 geocentric coordinates define the position of a point with 
///          respect to the center of mass of the earth.  The origin of the 
///          coordinate system is at the center of the WGS84 reference 
///          ellipsoid.  The positive X axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the prime meridian (0 degrees 
///          longitude), the positive Y axis intersects the ellipsoid at the 
///          equator (0 degrees latitude) on the 90th meridian east (90 degrees 
///          longitude), and the positive Z axis intersects the ellipsoid at 
///          the north pole.  This is an earth-centered, earth-fixed (ECEF) 
///          coordinate system.
///
/// \param   originLatitude  WGS84 geodetic latitude of the local origin in 
///                          radians.
/// \param   originLongitude WGS84 geodetic longitude of the local origin in 
///                          radians.
////////////////////////////////////////////////////////////////////////////////
////BLO_API const SmallMatrix<double, 3, 3> 
//    rotationFromWGS84LocalCartesianToWGS84Geocentric( double originLatitude, 
//    double originLongitude );

const Eigen::Matrix3d
rotationFromWGS84LocalCartesianToWGS84Geocentric(double originLatitude,
    double originLongitude);


//} // namespace blo

#endif // _BLO_GEOSPATIAL_H_