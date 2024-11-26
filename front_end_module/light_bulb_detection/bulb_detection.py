import numpy as np
import cv2

class LightSourceDetector:
    def __init__(self, filterByColor = True, blobColor = 255, filterByArea = True, minArea = 70, maxArea=8000,
                 filterByCircularity = False, filterByConvexity = False, filterByInertia = False):
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

    def connected_compnonent_analysis(self, dilated_img):
        output = cv2.connectedComponentsWithStats(dilated_img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # mask = np.zeros(dilated_img.shape, dtype="uint8")

        allMasks = []

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            widthKeep = 9 < w and 89 > w
            heightKeep = 9 < h and 89 > h
            areaKeep = 100 < area and area < 1400
            w_h_ratio = (w/h < 1.6) and (w/h > 0.7)

            componentMask = (labels == i).astype("uint8") * 255

            if all((widthKeep, heightKeep, areaKeep)):
                allMasks.append(componentMask)
            #print("Connected Component: " + str(i) + ") Height:", h, "Width:", w, "Area:", area)
        # axarr[i-1].imshow(componentMask , cmap='gray')
        # axarr[i-1].axis('off')
        # axarr[i-1].set_title("Connected Component: " + str(i))

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
        #imgBGRTest = cv2.imread(img)
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        binImage = cv2.inRange(YCbCr, np.array([y_channel[0], Cb_channel[0], Cr_channel[0]]),
                                    np.array([y_channel[1], Cb_channel[1], Cr_channel[1]]))

        kernelE = (3,3)
        kernel_dilation = (7,7)
        dilatedImg = self.morphological_processing(binImage, kernelE, kernel_dilation)

        blob_img = self.connected_compnonent_analysis(dilatedImg)

        keypoints = self.light_detector.detect(blob_img)

        return keypoints #, imgBGRTest
    