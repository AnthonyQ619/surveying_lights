import numpy as np
import cv2 as cv

class CameraPoseEstimator():
    def __init__(self, cameraMatrices, distortionCoeffs, stereoExtrinsics, maxDistance = 60.0):
        self.cam_left = cameraMatrices[0]
        self.cam_right = cameraMatrices[1]
        self.dist_left = distortionCoeffs[0]
        self.dist_right = distortionCoeffs[1]
        self.extrinsics = stereoExtrinsics
        self.R = stereoExtrinsics[:,:3]
        self.T = stereoExtrinsics[:, 3:]
        
        # Scene_points[pt_num] = X_i
        self.scene_points = []
        # # scene_point_2d_map[pt_num] = {frame_num : 2d_pt, etc..}
        # self.scene_point_2d_map = {}
        # # Scene Point to 2D mapping through bytes for fast comparisons
        # self.scene_point_2d_byte_map = {}

        # scene_points_map[pt_num] = [{pt.tobytes() : frame_num, ...}]
        self.scene_points_map = []

        # scene_points_2d_pts[pt_num] = [{frame_num : [left_2d_pt, right_2d_pt]}]
        self.scene_2d_pts = []

        # camera poses of left camera (relative to previous frame)
        self.rel_camera_poses = [np.hstack((np.eye(3), np.zeros((3,1))))]

        # camera_poses[frame_num_idx] = camera_pose(left and right)
        self.camera_poses = [[np.hstack((np.eye(3), np.zeros((3,1)))), stereoExtrinsics]]
    
    # Triangulation of points
    def triangulate_points(self, pts1, pts2, frame):
        xU1 = cv.undistortPoints(pts1, self.cam_left, self.dist_left)
        xU2 = cv.undistortPoints(pts2, self.cam_right, self.dist_right)

        P1mtx = np.eye(3) @ self.camera_poses[frame][0]
        P2mtx = np.eye(3) @ self.camera_poses[frame][1]

        X = cv.triangulatePoints(P1mtx, P2mtx, xU1, xU2)
        X = (X[:-1]/X[-1]).T

        return X

    def triangulate_nView_points(self, pt_index):

        # total_cameras = len(self.scene_point_2d_map[pt_index])
        total_cameras = len(self.scene_2d_pts[pt_index])
        A = np.zeros((4*total_cameras, 4))

        index = 0
        # Read Hartley and Zisserman to see if we need the normalization factor??
        for cam, pt in self.scene_2d_pts[pt_index].items():

            PmatLeft = np.eye(3) @ self.camera_poses[cam][0]
            PmatRight = np.eye(3) @ self.camera_poses[cam][1]

            xU1 = cv.undistortPoints(np.hstack(pt[0]), self.cam_left, self.dist_left)
            xU2 = cv.undistortPoints(np.hstack(pt[1]), self.cam_right, self.dist_right)

            row1 = xU1[0, 0, 0]*PmatLeft[2, :] - PmatLeft[0, :]
            row2 = xU1[0, 0, 1]*PmatLeft[2, :] - PmatLeft[1, :]
            row3 = xU2[0, 0, 0]*PmatRight[2, :] - PmatRight[0, :]
            row4 = xU2[0, 0, 1]*PmatRight[2, :] - PmatRight[1, :]

            A[4*index, :] = row1
            A[4*index + 1, :] = row2
            A[4*index + 2, :] = row3
            A[4*index + 3, :] = row4

        index += 1
        U, S, V = np.linalg.svd(A)
        X = V[-1, :]
        X = (X[:-1]/X[-1]).T

        return X