from __future__ import print_function
import cv2 as cv
import numpy as np
from scipy.sparse import lil_matrix

class BundleAdjustment():

    def __init__(self, K1, K2, distL, distR, extrinsics):
        self.KL = K1
        self.KR = K2
        self.distL = distL
        self.distR = distR
        self.R12_static = extrinsics[:, :3]
        self.T12_static = extrinsics[:, 3:].reshape((3,1))

    def project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""

        ws = camera_params[:, :3]
        trans = camera_params[:, 3:]

        points_proj = np.array([[0,0,0,0]])
        for par in range(ws.shape[0]):
            w = ws[par,:]
            w = w.reshape((3,1))

            Rot_L = cv.Rodrigues(w)[0]
            Trans_L  = trans[par,:].reshape((3,1))

            Rot_R = self.R12_static @ Rot_L
            Trans_R = self.R12_static @ Trans_L + self.T12_static

            x_left = cv.projectPoints(points[par, :], Rot_L, Trans_L, self.KL, self.distL)[0][0]
            x_right = cv.projectPoints(points[par, :], Rot_R, Trans_R, self.KR, self.distR)[0][0]

            estM = np.hstack((x_left, x_right))
            points_proj = np.vstack((points_proj, estM))


        return points_proj[1:, :]

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        # print("SHAPE est", points_proj.shape)
        # print("SHAPE GIVEN", points_2d.shape)
        #print(np.linalg.norm(points_proj - points_2d, axis=1))
        return (np.linalg.norm(points_proj - points_2d, axis=1)).ravel()
        # return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        #print(m)
        #print(len(i))
        for s in range(6):
            # If doing least square way!
            # A[4 * i, camera_indices * 6 + s] = 1
            # A[4 * i + 1, camera_indices * 6 + s] = 1
            # A[4 * i + 2, camera_indices * 6 + s] = 1
            # A[4 * i + 3, camera_indices * 6 + s] = 1
            # Doing distance based 
            A[i, camera_indices * 6 + s] = 1
            # A[i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            # Distance based 
            A[i, n_cameras * 6 + point_indices * 3 + s] = 1

            # A[i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

            # If doing least_square way
            # A[4 * i, camera_indices * 6 + point_indices * 3 + s] = 1
            # A[4 * i + 1, camera_indices * 6 + point_indices * 3 + s] = 1
            # A[4 * i + 2, camera_indices * 6 + point_indices * 3 + s] = 1
            # A[4 * i + 3, camera_indices * 6 + point_indices * 3 + s] = 1

        return A