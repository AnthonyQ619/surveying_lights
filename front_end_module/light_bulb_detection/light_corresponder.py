import cv2
import numpy as np

class LightSourceCorrespondenceDetector():
    def __init__(self, stereo_res, ids_res):

        # Math here to determine min/max blob size 
        self.stereo_res = stereo_res
        self.ids_res = ids_res

        self.max_difference = 950
        self.max_Xpixel_disparity = 30
        self.max_Ypixel_disparity = 60
        self.matchingList = None

    def det_match_stereo(self, kpL_pt, kpR_pt):
        if (abs(kpR_pt.size - kpL_pt.size) < self.max_difference and
            (abs(kpR_pt.pt[0] - kpL_pt.pt[0]) < self.max_Xpixel_disparity and
            abs(kpR_pt.pt[1] - kpL_pt.pt[1]) < self.max_Ypixel_disparity)):
            return True
        return False

    def blob_matcher_stereo(self, kpL, kpR):
        self.matchingList = []

        for i in range(len(kpL)):
            for j in range(len(kpR)):
                if self.det_match_stereo(kpL[i], kpR[j]):
                    self.matchingList.append((kpL[i], kpR[j]))


    def det_match_ids_cam(self, ids_kp):
        if self.matchingList is None:
            print("Must call Triangulate Matching for proper pipeline!")
            return 
        self.tri_match = []

        for i in range(len(self.matchingList)):
            right_kp = self.matchingList[i][1]
            x_pt, y_pt = right_kp.pt[0], right_kp.pt[1]
            roi_x = [max(0, x_pt - 30), min(x_pt + 30, self.ids_res[0])]
            roi_y = [max(0, y_pt - 10), min(y_pt + 90, self.ids_res[1])]
            for j in range(len(ids_kp)):
                ids_x_pt, ids_y_pt = ids_kp[j].pt[0], ids_kp[j].pt[0]
                if  (((ids_x_pt < roi_x[1]) and (ids_x_pt > roi_x[0])) and 
                    ((ids_y_pt < roi_y[1]) and (ids_y_pt > roi_x[1]))):
                    if abs(right_kp.size - ids_kp[j].size) < self.max_difference:
                        self.tri_match.append({"left": self.matchingList[i][0],
                                               "right": self.matchingList[i][1],
                                               "ids":ids_kp[j]})
        
    def tri_cam_matching(self, left_kp, right_kp, ids_kp):
        self.blob_matcher_stereo(left_kp, right_kp)
        self.det_match_ids_cam(ids_kp)

        return self.tri_match
