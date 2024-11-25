#!/usr/bin/python
import socket
import cv2
from picamera2 import Picamera2, Preview
import time
import numpy as np
from datetime import datetime
import sys

def build_img_buf_single(img1, img2, t):
    
    time_buf = t.encode()
    time_buf_size = str(len(t)).zfill(2).encode()
    img1_data = cv2.imencode('.png', img1)[1].tobytes()
    img1_bufsize = str(len(img1_data)).encode()
    img1_bufCountsize = str(len(str(len(img1_data)))).zfill(2).encode()

    img2_data = cv2.imencode('.png', img2)[1].tobytes()
    img2_bufsize = str(len(img2_data)).encode()
    img2_bufCountsize = str(len(str(len(img2_data)))).zfill(2).encode()
        
    _msg = time_buf_size + time_buf + img1_bufCountsize + img1_bufsize + img1_data + img2_bufCountsize + img2_bufsize + img2_data

    return _msg

def build_img_buf_stitched(stitched_image, t):
    
    time_buf = t.encode()
    time_buf_size = str(len(t)).zfill(2).encode()
    img1_data = cv2.imencode('.png', stitched_image)[1].tobytes()
    img1_bufsize = str(len(img1_data)).encode()
    img1_bufCountsize = str(len(str(len(img1_data)))).zfill(2).encode()
        
    _msg = time_buf_size + time_buf + img1_bufCountsize + img1_bufsize + img1_data

    return _msg

width = int(sys.argv[1])
height = int(sys.argv[2])
cam_exposure = int(sys.argv[3])
cam_gain = float(sys.argv[4])
framerate = int(sys.argv[5])

hostname = socket.gethostname()
print(hostname)
TCP_IP_1 = socket.gethostbyname(hostname)
print(TCP_IP_1)

TCP_IP = sys.argv[6] #'192.168.3.5'
TCP_PORT = 5001

print(TCP_IP)
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

sock.send("Acknowledge".encode())
time.sleep(1)


picam2a = Picamera2(0) 
picam2b = Picamera2(1) 
capture_configa = picam2a.create_still_configuration(main={"size": (width, height)})
capture_configb = picam2b.create_still_configuration(main={"size": (width, height)})
picam2a.configure(capture_configa) 
picam2b.configure(capture_configb)

minExp = picam2a.camera_controls['ExposureTime'][0]
maxExp = picam2a.camera_controls['ExposureTime'][1]

if cam_gain >= minExp and cam_gain <= maxExp:
    picam2a.set_controls({"AeEnable" : False, "AnalogueGain": cam_gain, "ExposureTime":  cam_exposure})
    picam2b.set_controls({"AeEnable" : False, "AnalogueGain": cam_gain, "ExposureTime":  cam_exposure})

picam2a.set_controls({"AeEnable" : False, "AnalogueGain": 6.0, "ExposureTime":  20000})
picam2b.set_controls({"AeEnable" : False, "AnalogueGain": 6.0, "ExposureTime":  20000})
time.sleep(0.5)
picam2a.start() 
picam2b.start()

time.sleep(1) 


count = float(sys.argv[4])

t1 = time.time()
frame = 0
frame_count = 0
while True:
    array1 = picam2a.capture_array("main") 
    array2 = picam2b.capture_array("main") 
    t = str(time.time())
    print(t)
    img1 = cv2.cvtColor(array1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(array2, cv2.COLOR_RGB2BGR)

    t2 = time.time()
    time_diff = abs(t1-t2)
    
    stitched_image = np.concatenate((img1, img2), axis=1)
    print(stitched_image.shape)
    #_msg = build_img_buf(img1, img2, t)
    _msg = build_img_buf_stitched(stitched_image, t)
    
    sock.send(_msg)
    frame += 1
    if time_diff > count:
        frame_count += 1
        print("Avg. Framerate:", frame/time_diff, "With Frame Count:", frame_count)
        frame = 0
        t1 = t2

