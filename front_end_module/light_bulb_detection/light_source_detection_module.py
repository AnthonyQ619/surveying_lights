import sys
import yaml
import cv2
import numpy as np
from bulb_detection import LightSourceDetector, BulbSourceAnalyzer
from light_corresponder import LightSourceCorrespondenceDetector

def video_reader_helper(video_file, n_frames = 80):
    cap = cv2.VideoCapture(video_file)
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        arr = np.array(gray_frame)
        all.append(arr)
        i += 1
    return np.array(all)

def main():
    video_path = "C:\\Users\\Anthony\\Documents\\Projects\\LCG_Experiments\\BRF_video_data\\Experiment2\\"
    image_path = "C:\\Users\\Anthony\\Documents\\Projects\\surveying_lights\\experiments\\exp7\\images\\image_pairs\\"

    test_index = 6

    vid_data = video_reader_helper(video_path + f"f{test_index}.avi")


    with open("light_bulb_calibrator\\bulb_constraints.yaml") as stream:
        try:
            light_constraints = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(light_constraints)
    constraints = light_constraints['Amber']

    c_y = [constraints['Y']['MIN'], constraints['Y']['MAX']]
    c_cb = [constraints['Cb']['MIN'], constraints['Cr']['MAX']]
    c_cr = [constraints['Cr']['MIN'], constraints['Cb']['MAX']]

    light_detector = LightSourceDetector(minArea = 30, maxArea=800, debug = True)

    keypoints_l, imgRGB_left = light_detector. detect_lightSource_YCbCr(image_path + f'left_camera\\frame{test_index}.png', 
                                                                c_y, c_cb, c_cr)
    
    keypoints_r, imgRGB_right = light_detector. detect_lightSource_YCbCr(image_path + f'right_camera\\frame{test_index}.png', 
                                                                c_y, c_cb, c_cr)
    
    drawn_img_l = cv2.drawKeypoints(imgRGB_left, keypoints_l, 0, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    drawn_img_r = cv2.drawKeypoints(imgRGB_right, keypoints_r, 0, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ld = BulbSourceAnalyzer(maxArea=150000, minArea=200, brf_filter=False, debug = False)
    # ld = LightSourceDetector(maxArea=150000, minArea=100, brf_filter=True

    kp, brf_points, brf_roi = ld.detect_lightSources(vid_data[60])

    im_with_keypoints = cv2.drawKeypoints(vid_data[60], kp, 0, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Frame Bulbs Left Image", drawn_img_l)
    cv2.imshow("Frame Bulbs Right Image", drawn_img_r)
    cv2.imshow("Frame Bulbs IR", im_with_keypoints)
    cv2.waitKey()
    print(vid_data.shape)


main()


