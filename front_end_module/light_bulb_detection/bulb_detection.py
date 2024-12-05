import numpy as np
import cv2
from operator import itemgetter
import matplotlib.pyplot as plt

class LightSourceDetector:
    def __init__(self, filterByColor = True, blobColor = 255, filterByArea = True, minArea = 70, maxArea=8000,
                 filterByCircularity = False, filterByConvexity = False, filterByInertia = False, debug = False):
        params = cv2.SimpleBlobDetector.Params()
        #params.maxThreshold = 255
        params.filterByColor = filterByColor
        params.blobColor = blobColor
        params.filterByArea = filterByArea
        #params.maxCircularity = 2500
        params.minArea = minArea
        params.maxArea = maxArea
        params.filterByCircularity = filterByCircularity
        params.filterByConvexity = filterByConvexity
        params.filterByInertia = filterByInertia

        self.light_detector = cv2.SimpleBlobDetector_create(params)
        self.minArea = minArea
        self.maxArea = maxArea
        self.debug = debug

    def connected_compnonent_analysis(self, dilated_img):
        output = cv2.connectedComponentsWithStats(dilated_img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        f, axarr = plt.subplots(numLabels-1, 1, tight_layout=True, figsize=(25,25))
        # mask = np.zeros(dilated_img.shape, dtype="uint8")

        allMasks = []

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            # widthKeep = 9 < w and 89 > w
            # heightKeep = 9 < h and 89 > h
            areaKeep = self.minArea < area and area < self.maxArea
            w_h_ratio = (w/h < 1.6) and (w/h > 0.5)

            componentMask = (labels == i).astype("uint8") * 255

            if all((w_h_ratio, areaKeep)):
                allMasks.append(componentMask)

            if self.debug:
              print("Connected Component: " + str(i) + ") Height:", h, "Width:", w, "Area:", area)
              axarr[i-1].imshow(componentMask, cmap='gray')
              axarr[i-1].axis('off')
              axarr[i-1].set_title("Connected Component: " + str(i))
        
        plt.show()

        # Blobs configured for light sources
        imageBlobs = np.zeros(dilated_img.shape, dtype="uint8")

        for imgMask in allMasks:
            imageBlobs += imgMask

        # cv2_imshow(imageBlobs)

        return imageBlobs

    def morphological_processing(self, binaryImage, kernelSize, kernel_dilation):
        kernel = np.ones(kernelSize, np.uint8)
        opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)

        kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        erosion = cv2.erode(opening, kernelE, iterations=1)

        kernelD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dilation)
        dilated_img = cv2.dilate(erosion,kernelD,iterations = 1)
        #cv2_imshow(dilated_img)

        return dilated_img

    def detect_lightSource_YCbCr(self, img, y_channel, Cb_channel, Cr_channel):
        imgBGRTest = cv2.imread(img)
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        YCbCr = cv2.cvtColor(imgBGRTest, cv2.COLOR_BGR2YCrCb)

        binImage = cv2.inRange(YCbCr, np.array([y_channel[0], Cb_channel[0], Cr_channel[0]]),
                                    np.array([y_channel[1], Cb_channel[1], Cr_channel[1]]))

        kernelE = (3,3)
        kernel_dilation = (7,7)
        dilatedImg = self.morphological_processing(binImage, kernelE, kernel_dilation)

        blob_img = self.connected_compnonent_analysis(dilatedImg)

        keypoints = self.light_detector.detect(blob_img)

        if self.debug:
            return keypoints, imgBGRTest
        else:
            return keypoints
    

class BulbSourceAnalyzer:
    def __init__(self, filterByColor = True, blobColor = 255, filterByArea = True, minArea = 70, maxArea=8000,
                 filterByCircularity = False, filterByConvexity = False, filterByInertia = False, brf_filter=True,
                 debug = False):
        params = cv2.SimpleBlobDetector.Params()
        params.filterByColor = filterByColor
        params.blobColor = blobColor
        params.filterByArea = filterByArea
        params.minArea = minArea
        params.maxArea = maxArea
        params.filterByCircularity = filterByCircularity
        params.filterByConvexity = filterByConvexity
        params.filterByInertia = filterByInertia

        self.light_detector = cv2.SimpleBlobDetector_create(params)
        self.brf_filter = brf_filter
        self.debug = debug
        self.minArea = minArea
        self.maxArea = maxArea

    def connected_compnonent_analysis(self, dilated_img):
        output = cv2.connectedComponentsWithStats(dilated_img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # mask = np.zeros(dilated_img.shape, dtype="uint8")
        f, axarr = plt.subplots(numLabels-1, 1, tight_layout=True, figsize=(25,25))
        allMasks = []
        cc_sizes = []

        # For Brf Analaysis only
        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            print(area)
            cc_sizes.append((area, i))

        sorted_cc_sizes = sorted(cc_sizes, key=itemgetter(0), reverse=True)

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            areaKeep = self.minArea < area and area < self.maxArea
            w_h_ratio = (w/h < 1.6) and (w/h > 0.5)

            componentMask = (labels == i).astype("uint8") * 255
            if self.brf_filter:
              print((labels == sorted_cc_sizes[0][1]).shape)
              componentMask = (labels == sorted_cc_sizes[0][1]).astype("uint8") * 255
              allMasks.append(componentMask)
              break
            else:
              if all((areaKeep, w_h_ratio)):
                allMasks.append(componentMask)

            if self.debug:
              print("Connected Component: " + str(i) + ") Height:", h, "Width:", w, "Area:", area)
              axarr[i-1].imshow(componentMask, cmap='gray')
              axarr[i-1].axis('off')
              axarr[i-1].set_title("Connected Component: " + str(i))
              plt.show()

        # Blobs configured for light sources
        imageBlobs = np.zeros(dilated_img.shape, dtype="uint8")

        for imgMask in allMasks:
            imageBlobs += imgMask

        return imageBlobs

    def morphological_processing(self, binaryImage, kernelSize, kernel_dilation):
        kernel = np.ones(kernelSize, np.uint8)
        opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)

        kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        erosion = cv2.erode(opening, kernelE, iterations=1)

        kernelD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dilation)
        dilated_img = cv2.dilate(erosion,kernelD,iterations = 1)

        return dilated_img
    
    def estimate_corner_points(self, img_edge, full_search=True):
        poi = []

        if full_search:
          for y in range(img_edge.shape[0]):
            for x in range(img_edge.shape[1]):
              if img_edge[y,x] > 0:
                  poi.append((x, y))
        else:
          y = img_edge.shape[0] // 2
          for x in range(img_edge.shape[0]):
              if img_edge[y, x] > 0:
                return (x, y)
        return poi
    
    def detect_lightSources(self, video_data, n_frame = 60):
        img = video_data[n_frame]

        # Fixed Kernel 
        kernel = (9, 9)

        blurred_img = cv2.GaussianBlur(img, kernel, 0)
        _, binImage = cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernelE = (3,3)
        kernel_dilation = (7,7)
        dilatedImg = self.morphological_processing(binImage, kernelE, kernel_dilation)

        # Image for debugging
        blob_img = self.connected_compnonent_analysis(dilatedImg)

        # Key Points used for matching between the cameras
        keypoints = self.light_detector.detect(blob_img)

        # Points used for analyzing BRF images (Spliced images from ROI points)
        points_for_roi = []
        pt_for_anlys = []
        for point in keypoints:
          x, y = point.pt[:]
          rad = point.size / 2

          topL_pt = (int(x - rad) - 5, int(y - rad) - 5)
          botR_pt = (int(x + rad) + 5, int(y + rad) + 5)

          blob_img_spliced = blob_img[topL_pt[1]:botR_pt[1], topL_pt[0]:botR_pt[0]]
          estimated_corners = cv2.Canny(blob_img_spliced, 100, 200)
          estimated_corners_pts = self.estimate_corner_points(estimated_corners, full_search=False)
          pt_for_anlys.append(estimated_corners_pts)
          points_for_roi.append((topL_pt, botR_pt))

        if self.debug:
            return keypoints, blob_img, pt_for_anlys, points_for_roi
        else:
            return keypoints, pt_for_anlys, points_for_roi