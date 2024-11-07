from picamera2 import Picamera2, Preview
import time 
import cv2
import numpy as np
import sys
import os

PATH = os.getcwd()

if not os.path.isdir(PATH + '/calibration_images'):
    os.mkdir(PATH + '/calibration_images')
IMAGE_PATH = PATH + '/calibration_images'

height = int(sys.argv[1])
width = int(sys.argv[2])

calib_dir = f'/calib_{height}_{width}'
if not os.path.isdir(IMAGE_PATH + calib_dir):
    os.mkdir(IMAGE_PATH + calib_dir)
IMAGE_PATH_CAL = IMAGE_PATH + calib_dir

picam2a = Picamera2(0) 
picam2b = Picamera2(1) 
capture_configa = picam2a.create_still_configuration(main={"size": (width, height)})
capture_configb = picam2b.create_still_configuration(main={"size": (width, height)})
picam2a.configure(capture_configa) 
picam2b.configure(capture_configb) 
picam2a.start() 
picam2b.start()

time.sleep(1) 


count = float(sys.argv[3])

t1 = time.time()
frame = 0
frame_count = 0
while True:
    array1 = picam2a.capture_array("main") 
    array2 = picam2b.capture_array("main") 
    
    img1 = cv2.cvtColor(array1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(array2, cv2.COLOR_RGB2BGR)

    t2 = time.time()
    time_diff = abs(t1-t2)
    
    stitched_image = np.concatenate((img1, img2), axis=1)

    stitched_image = np.concatenate((img1, img2), axis=1)
    if stitched_image.shape[1] > 1280:
        stitched_image = cv2.resize(stitched_image, (1280, 480), interpolation=cv2.INTER_AREA)

    cv2.imshow("Current Frame", stitched_image)
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
        cv2.imwrite(IMAGE_PATH_CAL + '/left_cam/' + f'frame{frame_count}.png', img1)
        cv2.imwrite(IMAGE_PATH_CAL + '/right_cam/' + f'frame{frame_count}.png', img2)
        frame = 0
        t1 = t2