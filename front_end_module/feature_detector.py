import cv2 as cv
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from structure_from_motion.structure_from_motion import CameraPoseEstimator
from structure_from_motion.bundle_adjustment import BundleAdjustment
from tqdm import tqdm
import time
from scipy.optimize import least_squares
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd

class NightTimeDetector():
    def __init__(self, ft, flann_params, clahe=True, night=True, debug = False):
        detector = {1: "sift", 2:"orb", 3:"brisk"}

        self.ft = detector[ft]
        self.night = night
        self.debug = debug

        if self.ft == "sift":
            self.ft_detect = cv.SIFT_create()

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = flann_params[0], trees = flann_params[1])
            #search_params = dict(checks=flann_params[2])
        elif self.ft == "orb":
            self.ft_detect = cv.ORB_create()

            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        elif self.ft == "brisk":
            self.ft_detect = cv.BRISK_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = flann_params[0], trees = flann_params[1])

        
        search_params = dict(checks=flann_params[2])
        self.index_params = index_params
        self.search_params = search_params
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        if clahe:
            self.clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))

    
    def preprocess_nightImage_hist(self, imgPath):
        img = cv.imread(imgPath)
        imgYCBCR = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        imgY = imgYCBCR[:,:,0]

        imgY = cv.equalizeHist(imgY)
        imgYCBCR[:,:,0] = imgY
        img = cv.cvtColor(imgYCBCR, cv.COLOR_YCrCb2BGR)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return imgGray, imgYCBCR

    def preprocess_nightImage_CLAHE(self, imgPath):
        img = cv.imread(imgPath)
        imgYCBCR = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        imgY = imgYCBCR[:,:,0]
        imgY_cl1 = self.clahe.apply(imgY)

        imgYCBCR[:,:,0] = imgY_cl1
        img = cv.cvtColor(imgYCBCR, cv.COLOR_YCrCb2BGR)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return imgGray, imgYCBCR


    def pre_process_image(self, imgFile):
        if self.night:
            if self.clahe:
                imgGray = self.preprocess_nightImage_CLAHE(imgFile)
            else:
                imgGray = self.preprocess_nightImage(imgFile)
        else:
            img = cv.imread(imgFile)
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #imgGray = cv.equalizeHist(imgGray)

        return imgGray

    def detect_features(self, imgFile):
        imgGray, imgYCBCR = self.pre_process_image(imgFile)

        kp, des = self.ft_detect.detectAndCompute(imgGray, None) #self.sift.detectAndCompute(imgGray, None)
        
        if self.debug:
            return kp, des, imgGray, imgYCBCR
        
        return kp, des

class NightTimeFeatureMatcher():

    def __init__(self, index_params, search_params, MaxDisparity = 50, k=2):
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        # self.bf = cv.BFMatcher(crossCheck=True)
        self.k = k 
        self.maxDisparity = MaxDisparity

    def match_keypoints(self, kpf1, kpf2):
        kp1, des1 = kpf1[:]
        kp2, des2 = kpf2[:]

        matches = self.flann.knnMatch(des1,des2,k=2)

        pts1 = []
        pts2 = []
        indicesL = []
        indicesR = []

        des1_match = []
        des2_match = []
        
        # ratio test as per Lowe's paper
        for match_i in matches:
            if len(match_i) == 2:
                m,n = match_i
                pix_disp = np.linalg.norm(np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt))
                #print(pix_disp)
                if m.distance < 0.7*n.distance and pix_disp <= self.maxDisparity: # Was 0.8, 0.75
                    #print("IN here")
                    # des2_match.append(des1[m.trainIdx])
                    # des1_match.append(des2[m.queryIdx])
                    pts2.append(kp2[m.trainIdx].pt)
                    indicesR.append(m.trainIdx)
                    pts1.append(kp1[m.queryIdx].pt)
                    indicesL.append(m.queryIdx)

        # matches = self.bf.knnMatch(np.array(des1_match), np.array(des2_match), k = self.k)
        # # Remove any cross checking case
        # matches = self.bf.knnMatch(np.array(des1_match), np.array(des2_match), k = self.k)
        # for match_i in matches:
        #     if len(match_i) == 2:
        #         m,n = match_i
        #         if m.distance < 0.7*n.distance:
        #             pts2.append(kp2[m.trainIdx].pt)
        #             indicesR.append(m.trainIdx)
        #             pts1.append(kp1[m.queryIdx].pt)
        #             indicesL.append(m.queryIdx)

        pts1 = np.float64(pts1)
        pts2 = np.float64(pts2)
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)

        if mask is None:
            return np.array([]), np.array([])

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        maskL = (mask.ravel()==1).tolist()
        indicesL = list(compress(indicesL, mask))
        indicesR = list(compress(indicesR, mask))


        kps1, des1 = np.array([kp1[x] for x in indicesL]), np.array([des1[x] for x in indicesL])
        kps2, des2 = np.array([kp2[x] for x in indicesR]), np.array([des2[x] for x in indicesR])

        return pts1.T, pts2.T, (kps1, des1), (kps2, des2)

    # Routine for Matching Key Points across frame[i] and frame[i+1]
    #  Returns indices for matched KeyPoints for frame[i] and frame[i+1]
    def match_keypoints_indices(self, kpf1, kpf2, MaxDisparity):
        kp1, des1 = kpf1[:]
        kp2, des2 = kpf2[:]

        matches = self.flann.knnMatch(des1,des2,k=2)

        #print("\n POINTS:", len(matches))

        pts1 = []
        pts2 = []
        indicesL = []
        indicesR = []

        # ratio test as per Lowe's paper
        # ratio test as per Lowe's paper
        for match_i in matches:
            if len(match_i) == 2:
                m,n = match_i
                pix_disp = np.linalg.norm(np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt))
                #print("Pixel disparity:", pix_disp)
                if m.distance < 0.8*n.distance and pix_disp <= self.maxDisparity: # Was 0.8, 0.75

                    #print("Pixel disparity:", np.linalg.norm(np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt)))
                    #print("IN here")
                    pts2.append(kp2[m.trainIdx].pt)
                    indicesR.append(m.trainIdx)
                    pts1.append(kp1[m.queryIdx].pt)
                    indicesL.append(m.queryIdx)

        # for i,(m,n) in enumerate(matches):
        #   if m.distance < 0.8*n.distance:
        #     pts2.append(kp2[m.trainIdx].pt)
        #     indicesR.append(m.trainIdx)
        #     pts1.append(kp1[m.queryIdx].pt)
        #     indicesL.append(m.queryIdx)


        ''' NEW ADDITION - SKIP FRAME'''
        if len(pts1) < 15 :
            return np.array([]), np.array([]), True

        pts1 = np.float64(pts1)
        pts2 = np.float64(pts2)
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, 3)

        if mask is None:
            return np.array([]), np.array([])
        maskL = (mask.ravel()==1).tolist()
        indicesL = list(compress(indicesL, mask))
        indicesR = list(compress(indicesR, mask))

        # SKIP FRAME ADDITION - ORIGINALLY: return np.array(indicesL), np.array(indicesR)
        return np.array(indicesL), np.array(indicesR), False

# General Helper Functions
def optimal_keypoint_index_matching(kp, des, indices):
    kp_opt, des_opt =   (np.array([kp[x] for x in indices]), 
                                np.array([des[x] for x in indices]))

    pts_opt = np.array([kp_opt[x].pt for x in range(kp_opt.shape[0])]).T
    
    new_kpts = []
    new_desp = []
    for i in range(len(kp)):
        if i not in indices:
            new_kpts.append(kp[i])
            new_desp.append(des[i])
    new_pts = (np.array(new_kpts), np.array(new_desp))


    return pts_opt, kp_opt, des_opt, new_pts 
 
import scipy.io as sio

def read_mat_calibration(filename):
  matlab_file = sio.loadmat(filename)

  intrinsicsData = []
  distortionData = []

  for i in range(len(matlab_file['calibration'][0][0]) - 1):
    if i < 2:
      intrinsicsData.append(matlab_file['calibration'][0][0][i])
    elif i < 4:
      distortionData.append(matlab_file['calibration'][0][0][i])
    else:
      R = matlab_file['calibration'][0][0][i]
      T = matlab_file['calibration'][0][0][i+1]/100 #Convert from Cm to Meters
      extrinsicsData = np.hstack((R, T.T))

  return intrinsicsData[0], intrinsicsData[1], distortionData[0], distortionData[1], extrinsicsData

def multiply_homog_matrices(mat1, mat2):
    mat1 = np.vstack((mat1, [[0,0,0,1]]))
    mat2 = np.vstack((mat2, [[0,0,0,1]]))

    r_mat = (mat1 @ mat2)[:3, :]

    return r_mat

def construct_point_cloud(point_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)

    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    window.add_child(scene)

    matGT = rendering.MaterialRecord()
    matGT.shader = 'defaultUnlit'
    matGT.point_size = 7.0
    matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

    scene.scene.add_geometry("mesh_name2", pcd, matGT)
    scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

    bounds = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    gui.Application.instance.run()

def covert_to_quat(r_Matrix):
    trace = np.trace(r_Matrix)

    m00 = r_Matrix[0,0]
    m01 = r_Matrix[0,1]
    m02 = r_Matrix[0,2]
    m10 = r_Matrix[1,0]
    m11 = r_Matrix[1,1]
    m12 = r_Matrix[1,2]
    m20 = r_Matrix[2,0]
    m21 = r_Matrix[2,1]
    m22 = r_Matrix[2,2]

    if (0 < trace):
        a = np.sqrt(trace + 1)
        b = 1 / (2 * a)

        qw = a / 2
        qx = (m21 - m12) * b
        qy = (m02 - m20) * b
        qz = (m10 - m01) * b

    else:
        i = 0
        if (m11 > m00): i = 1
        if (m22 > r_Matrix[i,i]): i = 2
        

        j = (i + 1) % 3
        k = (j + 1) % 3

        a = np.sqrt(max(0, r_Matrix[i,i] - r_Matrix[j,j] - r_Matrix[k,k] + 1))
        b = 1 / (2 * a)
 
        qw = (r_Matrix[k,j] - r_Matrix[j,k]) * b
        qx = a / 2
        qy = (r_Matrix[j,i] + r_Matrix[i,j]) * b
        qz = (r_Matrix[k,i] + r_Matrix[i,k]) * b
    # qw = np.sqrt(1 + m00 + m11 + m22) /2
    # qx = (m21 - m12)/( 4 *qw)
    # qy = (m02 - m20)/( 4 *qw)
    # qz = (m10 - m01)/( 4 *qw)

    return [qw, qx, qy, qz]

def main():
    exp_num = 8
    exp_ver = 1
    nt_ft = NightTimeDetector(1, [1, 5, 50], debug = True)
    nt_matcher = NightTimeFeatureMatcher(nt_ft.index_params, nt_ft.search_params, MaxDisparity=700)
    cam_left, cam_right, dist_left, dist_right, extrinsics = read_mat_calibration('C:\\Users\\Anthony\\Documents\\Projects\\surveying_lights\\calibration_script\\calibration_data\\calibrationData_640_480_GS_v3.mat')
    sfm_module = CameraPoseEstimator([cam_left, cam_right], [dist_left, dist_right], extrinsics)
    path = f"C:\\Users\Anthony\\Documents\\Projects\\surveying_lights\\experiments\\exp{exp_num}"
    left_img_path = path + "\\images\\image_pairs\\left_camera\\frame"
    right_img_path = path + "\\images\\image_pairs\\right_camera\\frame"
    frame = 120 # 80
    frame_count = 0 

    imgLeft = left_img_path + f"{frame}.png"
    imgRight = right_img_path + f"{frame}.png"

    imgL = cv.cvtColor(cv.imread(imgLeft),cv.COLOR_BGR2RGB)
    imgR = cv.cvtColor(cv.imread(imgRight),cv.COLOR_BGR2RGB)

    # Initial Point Tracking of the First Frame!
    kp_left, des_left, imgGray_left, imgYCBCR_L = nt_ft.detect_features(imgLeft)

    kp_right, des_right, imgGray_right, imgYCBCR_R = nt_ft.detect_features(imgRight)

    ptsL, ptsR, keyPointsL, keyPointsR = nt_matcher.match_keypoints((kp_left, des_left), (kp_right, des_right))
    kpL_opt, desL_opt = keyPointsL[0], keyPointsL[1]
    kpR_opt, desR_opt = keyPointsR[0], keyPointsR[1]
    

    for i in range(ptsL.shape[1]):
        sfm_module.scene_points_map.append({ptsL[:,i].tobytes() : frame_count})
        sfm_module.scene_2d_pts.append({frame_count : ([ptsL[:, i], ptsR[:, i]])})
    print("LENGTH OF SCENE POINT MAPPING", len(sfm_module.scene_points_map), "LENGTH OF 2D POINTS", len(sfm_module.scene_2d_pts))
    # Feature Detection of the following 2-N frames

    max_iter = 70
    for i in tqdm(range(max_iter)): # 59
        frame_count += 1
        #print(frame + frame_count)
        imgLeft_n = left_img_path + f"{frame + frame_count}.png"
        imgRight_n = right_img_path + f"{frame + frame_count}.png"
        # print(f"{frame + frame_count}.png")
        # Tracking feature points from current stereo pair to next individually
        f2_kpL, f2_desL, imgGray_left_n, imgYCBCR_L_n  = nt_ft.detect_features(imgLeft_n)
        indices_f1, indices_f2, skipFrame = nt_matcher.match_keypoints_indices([kpL_opt, desL_opt], [f2_kpL, f2_desL], 70)


        f2_kpR, f2_desR, imgGray_right_n, imgYCBCR_R_n  = nt_ft.detect_features(imgRight_n)
        indices_f1R, indices_f2R, skipFrame = nt_matcher.match_keypoints_indices([kpR_opt, desR_opt], [f2_kpR, f2_desR], 70)

        ptsL_T = ptsL[:, indices_f1] # Tracked points Only
        ptsR_T = ptsR[:, indices_f1R] # Tracked Points Only
        
        f2_pts_lft, f2_kpL_opt, f2_desL_opt, new_pts_L  = optimal_keypoint_index_matching(f2_kpL, f2_desL, indices_f2)
        f2_pts_R, f2_kpR_opt, f2_desR_opt, new_pts_R  = optimal_keypoint_index_matching(f2_kpR, f2_desR, indices_f2R)

        # print("Left Tracked Points", ptsL_T.shape, "Current Tracked Frame", f2_pts_lft.shape)
        # print("Right Tracked Points", ptsR_T.shape, "Current Tracked Frame", f2_pts_R.shape)


        # Stereo Matching of Tracked points (Utilized of relative motion estimates)
        # ptsL_n, ptsR_n, keyPointsL_n, keyPointsR_n = nt_matcher.match_keypoints_indices((f2_kpL_opt, f2_desL_opt), (f2_kpR_opt, f2_desR_opt), 50)
        indicesL, indicesR, skip = nt_matcher.match_keypoints_indices((f2_kpL_opt, f2_desL_opt), (f2_kpR_opt, f2_desR_opt), 50)

        # Tracked Points
        ptsL_n, ptsR_n = f2_pts_lft[:, indicesL], f2_pts_R[:, indicesR] 
        keyPointsL_n = (np.array([f2_kpL_opt[x] for x in indicesL]), np.array([f2_desL_opt[x] for x in indicesL]))
        keyPointsR_n = (np.array([f2_kpR_opt[x] for x in indicesR]), np.array([f2_desR_opt[x] for x in indicesR]))


        # Stereo Matching for the new points
        ptsL_new, ptsR_new, keyPointsL_new, keyPointsR_new = nt_matcher.match_keypoints((new_pts_L[0], new_pts_L[1]), (new_pts_R[0], new_pts_R[1])) # New Points

        #Update Scene Points
        ptsL_T_2 = ptsL_T[:, indicesL]
        ptsR_T_2 = ptsR_T[:, indicesR]
        # print(ptsL_T_2.shape)
        for k in range(len(sfm_module.scene_points_map)):
            for j in range(ptsL_T_2.shape[1]):
                if ptsL_T_2[:, j].tobytes() in sfm_module.scene_points_map[k]:
                    sfm_module.scene_points_map[k][ptsL_n[:, j].tobytes()] = frame_count
                    sfm_module.scene_2d_pts[k][frame_count] = (ptsL_n[:, j], ptsR_n[:, j])
                    break
            # print(j, ") Existing 2D points", len(sfm_module.scene_points[i]), "Frame to 2D point", len(scene_2d_pts[i]))
        
        # Add new Scene Points
        for j in range(ptsL_new.shape[1]):
            sfm_module.scene_points_map.append({ptsL_new[:,j].tobytes() : frame_count})
            sfm_module.scene_2d_pts.append({ frame_count : (ptsL_new[:,j], ptsR_new[:, j])})

        
        # TODO: Add NMS here comparing new points to tracked points

        # Now Estimate 3D tracked points from previous frame (Since it is only relative, we just use simple triangulation)

        X_3d = sfm_module.triangulate_points(ptsL_T_2, ptsR_T_2, 0)

        distances = np.linalg.norm(np.zeros(X_3d.shape) - X_3d, axis=1)
        #for i in range()
        # rmv_index = 
        # for j in range(distances.shape[0]):
        #     if distances[j] > 50:

        # print(distances.shape)
        # # print(X_3d.shape)
        # print(ptsL_n.shape)
        _, R_i, t_i = cv.solvePnP(X_3d, ptsL_n.T, cam_left, dist_left, flags=cv.SOLVEPNP_ITERATIVE)   #estimate_camera_pose(cam_left, dist_left, X, f2_pts_lft)
        R_i, _ = cv.Rodrigues(R_i)

        extL_i = np.hstack((R_i, np.vstack(t_i)))

        sfm_module.rel_camera_poses.append(extL_i)

        # Set current frame to previous frame data
        # print(ptsL.shape)
        # print(kpL_opt.shape)

        # print(ptsL_new.shape)
        # print(ptsL_n.shape)
        # print(np.hstack((ptsL_n, ptsL_new)).shape)

        # print(np.vstack(keyPointsL_new[0]).shape)
        # print(np.vstack(keyPointsL_n[0]).shape)
        # print(np.concatenate((keyPointsL_n[0], keyPointsL_new[0])).shape)

        ptsL, ptsR = np.hstack((ptsL_n, ptsL_new)), np.hstack((ptsR_n, ptsR_new))
        kpL_opt, desL_opt = (np.concatenate((keyPointsL_n[0], keyPointsL_new[0])),
                            np.concatenate((keyPointsL_n[1], keyPointsL_new[1])))
        kpR_opt, desR_opt = (np.concatenate((keyPointsR_n[0], keyPointsR_new[0])),
                            np.concatenate((keyPointsR_n[1], keyPointsR_new[1])))
        
        #extR_i = np.hstack((self.R@R_i, self.R@np.vstack(t_i) + self.T))
        # print(distances)
        # print(extL_i)
        # Add tracked points to previous points to 2D 
        # print(f2_pts_lft.shape)
        # print(ptsL_T.shape)
        # print(f2_pts_R.shape)
        # print("Total Points", len(f2_kpR), "New Points", len(new_pts_R[0]))
        # print(ptsR_T.shape)
    # f, ax = plt.subplots(3, 2, sharey=True, figsize=(14,8))
    print(sfm_module.rel_camera_poses)

    print(sfm_module.rel_camera_poses[1])
    print(sfm_module.rel_camera_poses[2])

    # Build compounding path of stereo cam
    print(sfm_module.R, sfm_module.T)
    for i in range(1, len(sfm_module.rel_camera_poses)):
        left_ext = multiply_homog_matrices(sfm_module.camera_poses[i- 1][0], sfm_module.rel_camera_poses[i])
        R_i, t_i = left_ext[:, :3], left_ext[:, 3:]
        right_ext = np.hstack((sfm_module.R@R_i, sfm_module.R@np.vstack(t_i) + sfm_module.T))
        sfm_module.camera_poses.append([left_ext, right_ext])


    print(sfm_module.camera_poses[-1])

    # Build 3D points
    for i in range(len(sfm_module.scene_points_map)):
        X = sfm_module.triangulate_nView_points(i)

        sfm_module.scene_points.append(X)

    print(len(sfm_module.scene_points))
    print(sfm_module.scene_points[0])


    # Items needed for Bundle Adjustment
    # x0, n_cameras, n_points, camera_ind, point_ind_new, points_2d_arr
    # n_cameras = number of cameras, where each element is the camera pose
    # n_points = number of 3D points, and their scene point value in X, Y, Z
    # camera_ind => The camera pose that belongs to each scene point
    # point_ind => the 2D points that belong to each Scene Point
    # points_2d => 2D corresponding points for each 3D scene point (Not ordered)

    # Cam Params N x 6 (R1, R2, R3, T1, T2, T3)
    cam_params = np.zeros((len(sfm_module.camera_poses), 6))
    for i in range(len(sfm_module.camera_poses)):
        R1, T1 = sfm_module.camera_poses[i][0][:, :3], sfm_module.camera_poses[i][0][:, 3:].reshape((1,3)) # Left camera only
        R1_rod = cv.Rodrigues(R1)[0].reshape((1,3))

        cam_params[i, :] = np.hstack((R1_rod, T1))

    # n_point construction (N, 3) for each 3D point
    points_3d = np.zeros((len(sfm_module.scene_points), 3))
    for i in range(len(sfm_module.scene_points)):
        X = sfm_module.scene_points[i]

        points_3d[i, :] = X.T
    
    cam_indices = []
    point_indices = []
    index = 0
    for i in range(len(sfm_module.scene_2d_pts)):
        for cam, points in sfm_module.scene_2d_pts[i].items():
            cam_indices.append(cam)
            point_indices.append(i)
    cam_indices = np.array(cam_indices)
    point_indices = np.array(point_indices)


    points_2d = np.zeros((cam_indices.shape[0], 4))

    for i in range(len(sfm_module.scene_2d_pts)):
        for _, points in sfm_module.scene_2d_pts[i].items():
            points_2d[index, :] = np.hstack((points[0], points[1]))
            index += 1

    print(cam_params.shape)
    print(points_3d.shape)
    print(cam_indices.shape)
    print(point_indices.shape)
    print(points_2d.shape)

    bun_adjuster = BundleAdjustment(cam_left, cam_right, dist_left, dist_right, extrinsics)
    x0 = np.hstack((cam_params.ravel(), points_3d.ravel()))
    n_cameras = cam_params.shape[0]
    n_points = points_3d.shape[0]

    f0 = bun_adjuster.fun(x0, n_cameras, n_points, cam_indices, point_indices, points_2d)

    plt.plot(f0)

    A = bun_adjuster.bundle_adjustment_sparsity(n_cameras, n_points, cam_indices, point_indices)

    print(f0.size)
    print(A.shape)
    print(x0.shape)
    n = 6 * n_cameras + 3 * n_points
    m = 4 * points_2d.shape[0]
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    print(n_cameras)
    print(n_points)

    res = least_squares(bun_adjuster.fun, x0, jac_sparsity=A, max_nfev=50, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, cam_indices, point_indices, points_2d))
    
    camera_data_df = pd.read_csv(path + f"\\images\\frames.csv")
    gps_data_df = pd.read_csv(path + f"\\gps_readings\\gps_service.csv")
    imu_data_df = pd.read_csv(path + f"\\imu_readings\\quat.csv")

    cam_poses = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d_opt = res.x[n_cameras * 6:].reshape((n_points, 3))

    # Optimized camera-poses from SfM/BA
    f_opt_poses = open(f'all_optimized_camera_poses_{exp_num}_{exp_ver}.csv', 'w')
    top_line = "timestamp,frame,"
    for i in range(0,4):
        top_line += "q" + str(i) + ","
    for i in range(0,3):
        top_line += "t" + str(i) + ","
    f_opt_poses.write(top_line[:-1] + '\n')

    starting_index = frame - 1
    for i in range(0, cam_poses.shape[0]):
        frame_num = starting_index
        timestamp = camera_data_df['Time'].iloc[frame_num]
        frame_c = camera_data_df['frame_num'].iloc[frame_num]
        rotL1 = cv.Rodrigues(cam_poses[i, :3])[0]
        transL1 = cam_poses[i, 3:].reshape((3,1))
        quaternion = covert_to_quat(rotL1)
        
        # temp_line = ""
        # for i in range(rotL1.shape[0]):
        #     temp_line += str(rotL1[i,0]) + ","
        # for i in range(0,3):
        #     temp_line += str(transL1[i,0]) + ","
        temp_line = f"{timestamp},{frame_c},"
        for i in range(0,4):
            temp_line += str(quaternion[i]) + ","
        for i in range(0,3):
            temp_line += str(transL1[i,0]) + ","
        
        f_opt_poses.write(temp_line[:-1] + '\n')
        starting_index += 1


    f_gps_pos = open(f'gps_pos_opt_{exp_num}_{exp_ver}.csv', 'w')
    top_line = "timestamp,longitude,latitude,altitude"
    f_gps_pos.write(top_line + '\n')

    f_imu_poses = open(f'imu_orientation_opt_{exp_num}_{exp_ver}.csv', 'w')
    top_line = "timestamp,"
    for i in range(0,4):
        top_line += "q" + str(i) + ","
    f_imu_poses.write(top_line[:-1] + '\n')

    gps_data_df
    imu_data_df
    count = 0
    proj_frame = 0
    #for index, row in imu_data_df.iterrows():
    print(imu_data_df.shape[0])
    for index in range(0, imu_data_df.shape[0]):
        count += 1
        if count == 41 and (proj_frame >= frame - 2 and proj_frame < (frame + max_iter) - 1):
            row = imu_data_df.iloc[index]
            # current reading
            timestamp = row["Time"]
            q0 = row["q1"]
            q1 = row["q2"]
            q2 = row["q3"]
            q3 = row["q4"]
            row_prev = imu_data_df.iloc[index - 1]

            timestamp_prev  = row_prev ["Time"]
            q0_prev  = row_prev ["q1"]
            q1_prev  = row_prev ["q2"]
            q2_prev  = row_prev ["q3"]
            q3_prev  = row_prev ["q4"]

            f_imu_poses.write(f"{timestamp_prev},{q0_prev},{q1_prev},{q2_prev},{q3_prev}\n")
            f_imu_poses.write(f"{timestamp},{q0},{q1},{q2},{q3}\n")
            
            count = 0
            proj_frame += 1
            # Previous  reading
        elif count == 41:
            count = 0
            proj_frame += 1
        

    count = 0
    proj_frame = 0
    for index in range(0, gps_data_df.shape[0]):
        count += 1
        if count == 41 and (proj_frame >= frame - 2 and proj_frame < (frame + max_iter) - 1):
            # current reading
            row = gps_data_df.iloc[index]
            timestamp = row["Time"]
            lon = row["Longitude"]
            lat = row["Latitude"]
            alt = row["Altitude"]
            
            row_prev = gps_data_df.iloc[index - 1]

            # Previous  reading
            timestamp_prev  = row_prev ["Time"]
            lon_prev = row["Longitude"]
            lat_prev = row["Latitude"]
            alt_prev = row["Altitude"]

            f_gps_pos.write(f"{timestamp_prev},{lon_prev},{lat_prev},{alt_prev}\n")
            f_gps_pos.write(f"{timestamp},{lon},{lat},{alt}\n")
            
            count = 0
            proj_frame += 1
        elif count == 41:
            count = 0
            proj_frame += 1
        
    # imgL = cv.cvtColor(imgYCBCR_L, cv.COLOR_YCrCb2RGB)
    # imgR = cv.cvtColor(imgYCBCR_R, cv.COLOR_YCrCb2RGB)
    # imgLn = cv.cvtColor(imgYCBCR_L_n, cv.COLOR_YCrCb2RGB)
    # imgRn = cv.cvtColor(imgYCBCR_R_n, cv.COLOR_YCrCb2RGB)

    # ax[0,0].imshow(imgL)
    # ax[0,1].imshow(imgLn)

    # ax[1,0].imshow(imgR)
    # ax[1,1].imshow(imgRn)

    # w = 9

    # # Tracked points between Left Image of N and N + 1
    # for i in range(ptsL_n.shape[1]):
    #     x1,y1 = ptsL_T_2 [:,i]
    #     x2,y2 = ptsL_n[:,i]
    #     ax[0,0].plot([x1, x2],[y1, y2],'-r')
    #     ax[0,0].add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
    #     ax[0,1].plot([x2, x1],[y2, y1],'-r')
    #     ax[0,1].add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))

    # # Tracked points between Right Image of N and N + 1
    # for i in range(ptsR_n.shape[1]):
    #     x1,y1 = ptsR_T_2[:,i]
    #     x2,y2 = ptsR_n[:,i]
    #     ax[1,0].plot([x1, x2],[y1, y2],'-r')
    #     ax[1,0].add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
    #     ax[1,1].plot([x2, x1],[y2, y1],'-r')
    #     ax[1,1].add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))
    construct_point_cloud(points_3d_opt)
    plt.plot(res.fun)
    # # Points 
    plt.show()

main()