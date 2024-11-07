from picamera2 import Picamera2, Preview
import time 
import cv2
import numpy as np
import sys
import os

PATH = os.getcwd()

if not os.path.isdir(PATH + '/calibration_images/single'):
    os.mkdir(PATH + '/calibration_images/single')
IMAGE_PATH = PATH + '/calibration_images/single'

height = int(sys.argv[1])
width = int(sys.argv[2])

calib_dir = f'/calib_{height}_{width}'
if not os.path.isdir(IMAGE_PATH + calib_dir):
    os.mkdir(IMAGE_PATH + calib_dir)
IMAGE_PATH_CAL = IMAGE_PATH + calib_dir

cam = int(sys.argv[3])
if cam == 0:
    cam_type = '/left_cam/'
else:
    cam_type = '/right_cam/'

picam2a = Picamera2(cam) 
capture_configa = picam2a.create_still_configuration(main={"size": (width, height)})
picam2a.configure(capture_configa) 
picam2a.start() 

time.sleep(1) 

count = float(sys.argv[4])

t1 = time.time()
frame = 0
frame_count = 0
while True:
    array1 = picam2a.capture_array("main") 
    
    img1 = cv2.cvtColor(array1, cv2.COLOR_RGB2BGR)

    t2 = time.time()
    time_diff = abs(t1-t2)
    
    cv2.imshow("img", img1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    frame += 1
    if time_diff > count:
        frame_count += 1
        print("Average framerate:", frame/count)
        print('COUNT:', frame_count)
        if not os.path.isdir(IMAGE_PATH_CAL + '/left_cam'):
            os.mkdir(IMAGE_PATH_CAL + '/left_cam')
        if not os.path.isdir(IMAGE_PATH_CAL + '/right_cam'):
            os.mkdir(IMAGE_PATH_CAL + '/right_cam')
            
        cv2.imwrite(IMAGE_PATH_CAL + cam_type + f'frame{frame_count}.png', img1)

        frame = 0
        t1 = t2