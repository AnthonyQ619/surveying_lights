////////////////////////////////////////////////////////////////////////////////
//                 Copyright (c) 2006-2021 Benjamin L. Ochoa
////////////////////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES
#include "geospatial.h"
#include <cmath>

#include <string>
#include <memory>
#include <cstdlib>
//#include <dtcc/CoordinateTuples/CartesianCoordinates.h>
#include <msp_dtcc/CartesianCoordinates.h>
#if defined( _MSC_VER )
#pragma warning( push )
// this function or variable may be unsafe
#pragma warning( disable: 4996 )
#endif // _MSC_VER
#include <msp_dtcc/CoordinateConversionException.h>
//#include <dtcc/Exception/CoordinateConversionException.h>
#if defined( _MSC_VER )
#pragma warning( pop )
#endif // _MSC_VER
#include <msp_dtcc/CoordinateSystem.h>
#include <msp_dtcc/Geocentric.h>
#include <msp_dtcc/GeodeticCoordinates.h>
#include <msp_dtcc/GeoidLibrary.h>
#include <msp_dtcc/LocalCartesian.h>
//#include <dtcc/CoordinateSystems/LocalCartesian.h>
//#include <dtcc/CoordinateSystems/CoordinateSystem.h>
//#include <dtcc/CoordinateSystems/Geocentric.h>
//#include <dtcc/CoordinateTuples/GeodeticCoordinates.h>
//#include <dtcc/CoordinateSystems/GeodeticCoordinates.h>
//#include <dtcc/GeoidLibrary.h>
#include <eigen3/Eigen/Dense>

#include "blo/Point.h"
#include "blo/Exception.h"

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::RowVector3d;
using Eigen::Vector3d;
using Eigen::Vector4d;

//namespace blo
//{
const char* geospatialDataDir()
{
#if defined( _MSC_VER )
#pragma warning( push )
// this function or variable may be unsafe
#pragma warning( disable: 4996 )
#endif // _MSC_VER
    return getenv( "MSPCCS_DATA" );
#if defined( _MSC_VER )
#pragma warning( pop )
#endif // _MSC_VER
}

void setGeospatialDataDir( const char* dataDir )
{
    if ( nullptr == dataDir )
    {
        throw Exception( Status::nullPtrError );
    }


    const std::string env( "MSPCCS_DATA=" + std::string( dataDir ) );
    if ( 0 != _putenv( env.c_str() ) )
    {
        throw Exception( Status::error );
    }
}
//} // namespace blo

namespace
{
// Deleter to use with unique_ptr<MSP::CCS::GeoidLibrary>
struct GeoidLibrary_removeInstance
{
    void operator()( MSP::CCS::GeoidLibrary* pGeoidLibrary ) const
    {
        pGeoidLibrary->removeInstance();
    }
};
} // namespace

//namespace blo
//{
double WGS84EllipsoidHeightToOrthometricHeight_EGM2008( 
    double latitude, double longitude, double ellipsoidHeight )
{
    double orthometricHeight;
    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary,
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );

        pGeoidLibrary->convertEllipsoidHeightToEGM2008GeoidHeight( longitude, 
            latitude, ellipsoidHeight, &orthometricHeight );
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }

    return orthometricHeight;
}

void WGS84EllipsoidHeightToOrthometricHeight_EGM2008( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcEllipsoidHeight, double* pDst, int count )
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
         nullptr == pSrcEllipsoidHeight || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );
        for ( int n = 0; n < count; ++n )
        {
            pGeoidLibrary->convertEllipsoidHeightToEGM2008GeoidHeight( 
                *pSrcLongitude, *pSrcLatitude, *pSrcEllipsoidHeight, pDst );

            ++pSrcLatitude;
            ++pSrcLongitude;
            ++pSrcEllipsoidHeight;
            ++pDst;
        }
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }
}

double orthometricHeightToWGS84EllipsoidHeight_EGM2008( 
    double latitude, double longitude, double orthometricHeight )
{
    double ellipsoidHeight;
    try
    {
        std::cout << "HERE" << std::endl;
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );

        std::cout << "HERE 2" << std::endl;
        pGeoidLibrary->convertEGM2008GeoidHeightToEllipsoidHeight( longitude, 
            latitude, orthometricHeight, &ellipsoidHeight );
        std::cout << "HERE 3" << std::endl;
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }

    return ellipsoidHeight;
}

void orthometricHeightToWGS84EllipsoidHeight_EGM2008( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcOrthometricHeight, double* pDst, int count )
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
         nullptr == pSrcOrthometricHeight || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );
        for ( int n = 0; n < count; ++n )
        {
            pGeoidLibrary->convertEGM2008GeoidHeightToEllipsoidHeight( 
                *pSrcLongitude, *pSrcLatitude, *pSrcOrthometricHeight, pDst );

            ++pSrcLatitude;
            ++pSrcLongitude;
            ++pSrcOrthometricHeight;
            ++pDst;
        }
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }
}

double WGS84EllipsoidHeightToOrthometricHeight_EGM96( double latitude, 
    double longitude, double ellipsoidHeight )
{
    double orthometricHeight;
    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );

        pGeoidLibrary->convertEllipsoidToEGM96FifteenMinBilinearGeoidHeight( 
            longitude, latitude, ellipsoidHeight, &orthometricHeight );
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }

    return orthometricHeight;
}

void WGS84EllipsoidHeightToOrthometricHeight_EGM96( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcEllipsoidHeight, double* pDst, int count )
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
         nullptr == pSrcEllipsoidHeight || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );
        for ( int n = 0; n < count; ++n )
        {
            pGeoidLibrary->
                convertEllipsoidToEGM96FifteenMinBilinearGeoidHeight( 
                *pSrcLongitude, *pSrcLatitude, *pSrcEllipsoidHeight, pDst );

            ++pSrcLatitude;
            ++pSrcLongitude;
            ++pSrcEllipsoidHeight;
            ++pDst;
        }
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }
}

double orthometricHeightToWGS84EllipsoidHeight_EGM96( double latitude, 
    double longitude, double orthometricHeight )
{
    double ellipsoidHeight;
    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );

        pGeoidLibrary->convertEGM96FifteenMinBilinearGeoidToEllipsoidHeight( 
            longitude, latitude, orthometricHeight, &ellipsoidHeight );
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }

    return ellipsoidHeight;
}

void orthometricHeightToWGS84EllipsoidHeight_EGM96( 
    const double* pSrcLatitude, const double* pSrcLongitude, 
    const double* pSrcOrthometricHeight, double* pDst, int count )
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
         nullptr == pSrcOrthometricHeight || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    try
    {
        static std::unique_ptr<MSP::CCS::GeoidLibrary, 
            GeoidLibrary_removeInstance> pGeoidLibrary( 
            MSP::CCS::GeoidLibrary::getInstance() );
        for ( int n = 0; n < count; ++n )
        {
            pGeoidLibrary->
                convertEGM96FifteenMinBilinearGeoidToEllipsoidHeight( 
                *pSrcLongitude, *pSrcLatitude, *pSrcOrthometricHeight, pDst );

            ++pSrcLatitude;
            ++pSrcLongitude;
            ++pSrcOrthometricHeight;
            ++pDst;
        }
    }
    catch ( MSP::CCS::CoordinateConversionException& e )
    {
        throw Exception( e.getMessage() );
    }
}
//} // namespace blo


/// Changes go here
///  Changed  blo::SmallMatrix<double, 4, 4> calcWGS84GeocentricQuadric() to 
/// Changes go here
namespace
{
    //Expect matrix of size 4x4
const Eigen::MatrixXd calcWGS84GeocentricQuadric()
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // Lengths of semi-axes in meters
    const double a = ellipsoidSemiMajorAxis;
    const double b = ellipsoidSemiMajorAxis;
    const double c = ellipsoidSemiMajorAxis * ( 1 - ellipsoidFlattening );

    // Quadric
    //     [ b^2 * c^2      0          0              0        ]
    // Q = [     0      a^2 * c^2      0              0        ]
    //     [     0          0      a^2 * b^2          0        ]
    //     [     0          0          0      -a^2 * b^2 * c^2 ]

    // Scale Q = diag(q) such that its norm is 1
    Eigen::Vector4d q;
    q(0) = -1 / ( a * a );
    q(1) = -1 / ( b * b );
    q(2) = -1 / ( c * c );
    q(3) = 1;

    return Eigen::MatrixXd(q.asDiagonal());
}
} // namespace

//namespace blo
//{
const Eigen::Matrix4d WGS84GeocentricQuadric()
{
    static const Eigen::Matrix4d Q = calcWGS84GeocentricQuadric();

    return Q;
}

const Eigen::Vector3d WGS84GeodeticToWGS84Geocentric( double latitude,
    double longitude, double ellipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 geocentric coordinate frame
    MSP::CCS::Geocentric geocentricWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening );

    // Convert from WGS84 geodetic coordinates to WGS84 geocentric coordinates
    const MSP::CCS::GeodeticCoordinates geodeticCoords( 
        MSP::CCS::CoordinateType::geodetic, longitude, latitude, 
        ellipsoidHeight );
    const MSP::CCS::CartesianCoordinates* pGeocentricCoords = 
        geocentricWGS84.convertFromGeodetic( &geodeticCoords );
    const Eigen::Vector3d ret{ pGeocentricCoords->x(), pGeocentricCoords->y(),
        pGeocentricCoords->z() };
    delete pGeocentricCoords;

    return ret;
}

// Rather than point class (pDst), treat points in a matrix (similar to python) and access it through Eigen
// MatrixXd allows points to be in 3d with any number of points -> 3 rows for 3D (X,Y,Z) and col = num of points
// Give a matrix of size 3xCount filled with zeros to properly fill the matrix.
void WGS84GeodeticToWGS84Geocentric( const double* pSrcLatitude, 
    const double* pSrcLongitude, const double* pSrcEllipsoidHeight, 
    Eigen::MatrixXd& pDst, int count ) //UPDATE HERE -> Change Point class and update it to fit what Ben is doing.
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
         nullptr == pSrcEllipsoidHeight) //|| nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 geocentric coordinate frame
    MSP::CCS::Geocentric geocentricWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening );

    // Convert from WGS84 geodetic coordinates to WGS84 geocentric coordinates
    MSP::CCS::GeodeticCoordinates geodeticCoords( 
        MSP::CCS::CoordinateType::geodetic );
    const MSP::CCS::CartesianCoordinates* pGeocentricCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        geodeticCoords.set( *pSrcLongitude, *pSrcLatitude, 
            *pSrcEllipsoidHeight );
        pGeocentricCoords = geocentricWGS84.convertFromGeodetic( 
            &geodeticCoords );
        /*pDst->setCoords( pGeocentricCoords->x(), pGeocentricCoords->y(), 
            pGeocentricCoords->z() );*/
        pDst(0, n) = pGeocentricCoords->x(); 
        pDst(1, n) = pGeocentricCoords->y();
        pDst(2, n) = pGeocentricCoords->z(); 
        delete pGeocentricCoords;

        ++pSrcLatitude;
        ++pSrcLongitude;
        ++pSrcEllipsoidHeight;
        //++pDst;
    }
}

void WGS84GeocentricToWGS84Geodetic( const Eigen::Vector3d& src,
    double& latitude, double& longitude, double& ellipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 geocentric coordinate frame
    MSP::CCS::Geocentric geocentricWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening );

    // Convert from WGS84 geocentric coordinates to WGS84 geodetic coordinates 
    // (iterative algorithm)
    const MSP::CCS::CartesianCoordinates geocentricCoords( 
        MSP::CCS::CoordinateType::geocentric, src(0), src(1), src(2));
    const MSP::CCS::GeodeticCoordinates* pGeodeticCoords = 
        geocentricWGS84.convertToGeodetic( 
        const_cast<MSP::CCS::CartesianCoordinates*>( &geocentricCoords ) );
    latitude = pGeodeticCoords->latitude();
    longitude = pGeodeticCoords->longitude();
    ellipsoidHeight = pGeodeticCoords->height();
    delete pGeodeticCoords;
}


// Rather than point class (pSrc), treat points in a matrix (similar to python) and access it through Eigen
// MatrixXd allows points to be in 3d with any number of points -> 3 rows for 3D (X,Y,Z) and col = num of points
void WGS84GeocentricToWGS84Geodetic( const Eigen::MatrixXd& pSrc,
    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight, 
    int count )
{
    /*if ( nullptr == pSrc || nullptr == pDstLatitude || 
         nullptr == pDstLongitude || nullptr == pDstEllipsoidHeight )*/
    if (nullptr == pDstLatitude || nullptr == pDstLongitude || 
        nullptr == pDstEllipsoidHeight)
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 geocentric coordinate frame
    MSP::CCS::Geocentric geocentricWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening );

    // Convert from WGS84 geocentric coordinates to WGS84 geodetic coordinates 
    // (iterative algorithm)
    MSP::CCS::CartesianCoordinates geocentricCoords( 
        MSP::CCS::CoordinateType::geocentric );
    const MSP::CCS::GeodeticCoordinates* pGeodeticCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        geocentricCoords.set( pSrc(0, n), pSrc(1, n), pSrc(2, n) );
        pGeodeticCoords = geocentricWGS84.convertToGeodetic( 
            &geocentricCoords );
        *pDstLatitude = pGeodeticCoords->latitude();
        *pDstLongitude = pGeodeticCoords->longitude();
        *pDstEllipsoidHeight = pGeodeticCoords->height();
        delete pGeodeticCoords;

        //++pSrc;
        ++pDstLatitude;
        ++pDstLongitude;
        ++pDstEllipsoidHeight;
    }
}


const Eigen::Vector3d WGS84GeodeticToWGS84LocalCartesian( 
    double latitude, double longitude, double ellipsoidHeight, 
    double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 geodetic coordinates to WGS84 local Cartesian 
    // coordinates
    const MSP::CCS::GeodeticCoordinates geodeticCoords( 
        MSP::CCS::CoordinateType::geodetic, longitude, latitude, 
        ellipsoidHeight );
    const MSP::CCS::CartesianCoordinates* pLocalCartesianCoords = 
        localCartesianWGS84.convertFromGeodetic( 
        const_cast<MSP::CCS::GeodeticCoordinates*>( &geodeticCoords ) );
    const Eigen::Vector3d ret{ {pLocalCartesianCoords->x(),
        pLocalCartesianCoords->y(), pLocalCartesianCoords->z()} };
    delete pLocalCartesianCoords;

    return ret;
}

// Rather than point class (pDst), treat points in a matrix (similar to python) and access it through Eigen
// MatrixXd allows points to be in 3d with any number of points -> 3 rows for 3D (X,Y,Z) and col = num of points
// Give a matrix of size 3xCount filled with zeros to properly fill the matrix.
void WGS84GeodeticToWGS84LocalCartesian( const double* pSrcLatitude, 
    const double* pSrcLongitude, const double* pSrcEllipsoidHeight, 
    Eigen::MatrixXd& pDst, int count, double originLatitude, 
    double originLongitude, double originEllipsoidHeight )
{
    if ( nullptr == pSrcLatitude || nullptr == pSrcLongitude || 
        nullptr == pSrcEllipsoidHeight) //|| nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 geodetic coordinates to WGS84 local Cartesian 
    // coordinates
    MSP::CCS::GeodeticCoordinates geodeticCoords( 
        MSP::CCS::CoordinateType::geodetic );
    const MSP::CCS::CartesianCoordinates* pLocalCartesianCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        geodeticCoords.set( *pSrcLongitude, *pSrcLatitude, 
            *pSrcEllipsoidHeight );
        pLocalCartesianCoords = localCartesianWGS84.convertFromGeodetic( 
            &geodeticCoords );
        /*pDst->setCoords( pLocalCartesianCoords->x(), 
            pLocalCartesianCoords->y(), pLocalCartesianCoords->z() );*/
        pDst(0, n) = pLocalCartesianCoords->x();
        pDst(1, n) = pLocalCartesianCoords->y();
        pDst(2, n) = pLocalCartesianCoords->z();
        delete pLocalCartesianCoords;

        ++pSrcLatitude;
        ++pSrcLongitude;
        ++pSrcEllipsoidHeight;
        //++pDst;
    }
}

void WGS84LocalCartesianToWGS84Geodetic( const Eigen::Vector3d& src, 
    double& latitude, double& longitude, double& ellipsoidHeight, 
    double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 local Cartesian coordinates to WGS84 geodetic 
    // coordinates (iterative algorithm)
    const MSP::CCS::CartesianCoordinates localCartesianCoords( 
        MSP::CCS::CoordinateType::localCartesian, src(0), src(1), src(2) );
    const MSP::CCS::GeodeticCoordinates* pGeodeticCoords = 
        localCartesianWGS84.convertToGeodetic( 
        const_cast<MSP::CCS::CartesianCoordinates*>( &localCartesianCoords ) );
    latitude = pGeodeticCoords->latitude();
    longitude = pGeodeticCoords->longitude();
    ellipsoidHeight = pGeodeticCoords->height();
    delete pGeodeticCoords;
}

// Rather than point class (pSrc), treat points in a matrix (similar to python) and access it through Eigen
// MatrixXd allows points to be in 3d with any number of points -> 3 rows for 3D (X,Y,Z) and col = num of points
void WGS84LocalCartesianToWGS84Geodetic( const Eigen::MatrixXd& pSrc, 
    double* pDstLatitude, double* pDstLongitude, double* pDstEllipsoidHeight, 
    int count, double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    /*if ( nullptr == pSrc || nullptr == pDstLatitude || 
         nullptr == pDstLongitude || nullptr == pDstEllipsoidHeight )
    {*/
    if (nullptr == pDstLatitude || nullptr == pDstLongitude || 
        nullptr == pDstEllipsoidHeight)
    {
        throw Exception( Status::nullPtrError );
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 local Cartesian coordinates to WGS84 geodetic 
    // coordinates (iterative algorithm)
    MSP::CCS::CartesianCoordinates localCartesianCoords( 
        MSP::CCS::CoordinateType::localCartesian );
    const MSP::CCS::GeodeticCoordinates* pGeodeticCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        localCartesianCoords.set( pSrc(0, n), pSrc(1, n), pSrc(2, n) );
        pGeodeticCoords = localCartesianWGS84.convertToGeodetic( 
            &localCartesianCoords );
        *pDstLatitude = pGeodeticCoords->latitude();
        *pDstLongitude = pGeodeticCoords->longitude();
        *pDstEllipsoidHeight = pGeodeticCoords->height();
        delete pGeodeticCoords;

        //++pSrc;
        ++pDstLatitude;
        ++pDstLongitude;
        ++pDstEllipsoidHeight;
    }
}

const Eigen::Vector3d WGS84GeocentricToWGS84LocalCartesian(
    const Eigen::Vector3d& src, double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 geocentric coordinates to WGS84 local Cartesian 
    // coordinates
    const MSP::CCS::CartesianCoordinates geocentricCoords( 
        MSP::CCS::CoordinateType::geocentric, src(0), src(1), src(2) );
    const MSP::CCS::CartesianCoordinates* pLocalCartesianCoords = 
        localCartesianWGS84.convertFromGeocentric( &geocentricCoords );
    const Eigen::Vector3d ret{ { pLocalCartesianCoords->x(),
        pLocalCartesianCoords->y(), pLocalCartesianCoords->z() } };
    delete pLocalCartesianCoords;

    return ret;
}

// Rather than point class (pDst/ pSrc), treat points in a matrix (similar to python) and access it through Eigen
// MatrixXd allows points to be in 3d with any number of points -> 3 rows for 3D (X,Y,Z) and col = num of points
// Give a matrix of size 3xCount filled with zeros to properly fill the matrix.
void WGS84GeocentricToWGS84LocalCartesian( 
    const Eigen::MatrixXd& pSrc, Eigen::MatrixXd& pDst, int count,
    double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    /*if ( nullptr == pSrc || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }*/
    if (pSrc.cols() != pDst.cols()){
        throw Exception(Status::sizeError);
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 geocentric coordinates to WGS84 local Cartesian 
    // coordinates
    MSP::CCS::CartesianCoordinates geocentricCoords( 
        MSP::CCS::CoordinateType::geocentric );
    const MSP::CCS::CartesianCoordinates* pLocalCartesianCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        geocentricCoords.set( pSrc(0, n), pSrc(1, n), pSrc(2, n));
        pLocalCartesianCoords = localCartesianWGS84.convertFromGeocentric( 
            &geocentricCoords );
        pDst(0, n) = pLocalCartesianCoords->x();
        pDst(1, n) = pLocalCartesianCoords->y();
        pDst(2, n) = pLocalCartesianCoords->z();
        delete pLocalCartesianCoords;

        //++pSrc;
        //++pDst;
    }
}

// Have in Rotation File
//const SmallMatrix<double, 3, 3> 
//    rotationFromWGS84GeocentricToWGS84LocalCartesian( double originLatitude, 
//    double originLongitude )
//{
//    using std::sin;
//    using std::cos;
//
//    const double sin_lat = sin( originLatitude );
//    const double cos_lat = cos( originLatitude );
//    const double sin_lon = sin( originLongitude );
//    const double cos_lon = cos( originLongitude );
//
//    SmallMatrix<double, 3, 3> R;
//    R[0][0] = -sin_lon;
//    R[0][1] =  cos_lon;
//    R[0][2] =  0;
//    R[1][0] = -sin_lat * cos_lon;
//    R[1][1] = -sin_lat * sin_lon;
//    R[1][2] =  cos_lat;
//    R[2][0] =  cos_lat * cos_lon;
//    R[2][1] =  cos_lat * sin_lon;
//    R[2][2] =  sin_lat;
//    return R;
//}

const Eigen::Vector3d WGS84LocalCartesianToWGS84Geocentric(
    const Eigen::Vector3d& src, double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 local Cartesian coordinates to WGS84 geocentric 
    // coordinates
    const MSP::CCS::CartesianCoordinates localCartesianCoords( 
        MSP::CCS::CoordinateType::localCartesian, src(0), src(1), src(2) );
    const MSP::CCS::CartesianCoordinates* pGeocentricCoords = 
        localCartesianWGS84.convertToGeocentric( &localCartesianCoords );
    const Eigen::Vector3d ret{ { pGeocentricCoords->x(), pGeocentricCoords->y(),
        pGeocentricCoords->z() } };
    delete pGeocentricCoords;

    return ret;
}

void WGS84LocalCartesianToWGS84Geocentric( 
    const Eigen::MatrixXd& pSrc, Eigen::MatrixXd& pDst, int count,
    double originLatitude, double originLongitude, 
    double originEllipsoidHeight )
{
    /*if ( nullptr == pSrc || nullptr == pDst )
    {
        throw Exception( Status::nullPtrError );
    }*/
    if (pSrc.cols() != pDst.cols())
    {
        throw Exception(Status::sizeError);
    }
    if ( 0 >= count )
    {
        throw Exception( Status::sizeError );
    }


    // WGS84 ellipsoid parameters
    MSP::CCS::CoordinateSystem coordinateSystemWGS84;
    double ellipsoidSemiMajorAxis, ellipsoidFlattening;
    coordinateSystemWGS84.getEllipsoidParameters( &ellipsoidSemiMajorAxis, 
        &ellipsoidFlattening );

    // WGS84 local Cartesian coordinate frame
    MSP::CCS::LocalCartesian localCartesianWGS84( ellipsoidSemiMajorAxis, 
        ellipsoidFlattening, originLongitude, originLatitude, 
        originEllipsoidHeight, 0 );

    // Convert from WGS84 local Cartesian coordinates to WGS84 geocentric 
    // coordinates
    MSP::CCS::CartesianCoordinates localCartesianCoords( 
        MSP::CCS::CoordinateType::localCartesian );
    const MSP::CCS::CartesianCoordinates* pGeocentricCoords = nullptr;
    for ( int n = 0; n < count; ++n )
    {
        localCartesianCoords.set( pSrc(0, n), pSrc(1, n), pSrc(2, n) );
        pGeocentricCoords = localCartesianWGS84.convertToGeocentric( 
            &localCartesianCoords );
        pDst(0, n) = pGeocentricCoords->x();
        pDst(1, n) = pGeocentricCoords->y();
        pDst(2, n) = pGeocentricCoords->z();
        delete pGeocentricCoords;

        // ++pSrc;
        // ++pDst;
    }
}

const Eigen::Matrix3d
    rotationFromWGS84LocalCartesianToWGS84Geocentric( double originLatitude, 
    double originLongitude )
{
    using std::sin;
    using std::cos;

    const double sin_lat = sin( originLatitude );
    const double cos_lat = cos( originLatitude );
    const double sin_lon = sin( originLongitude );
    const double cos_lon = cos( originLongitude );

    Eigen::Matrix3d R;
    R(0, 0) = -sin_lon;
    R(0, 1) = -sin_lat * cos_lon;
    R(0, 2) =  cos_lat * cos_lon;
    R(1, 0) =  cos_lon;
    R(1, 1) = -sin_lat * sin_lon;
    R(1, 2) =  cos_lat * sin_lon;
    R(2, 0) =  0;
    R(2, 1) =  cos_lat;
    R(2, 2) =  sin_lat;
    return R;
}

//} // namespace blo