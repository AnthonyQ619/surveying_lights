import numpy as np
import matplotlib.pyplot as plt
import cv2
from color_space_analysis import *
from roi_selection import *

def main():
  csv_file = "test_sample_rois_GS_1.csv"
  image_path = "C:\\Users\\Anthony\\Documents\\Projects\\surveying_lights\\experiments\\sample_data\\left_camera\\"

  # Read Data
  data = csv_roi_reader(csv_file, True)
  print(data)

  img_nums = list(data.keys())
  print(img_nums)

  roi_white = []
  roi_amber = []

  for i in range(len(img_nums)):
    if 'white' in data[img_nums[i]]:
      #print(image_path + 'frame'+str(img_nums[i])+'.png')
      roi_set_white = get_roi(image_path + 'frame'+str(img_nums[i])+'.png', data[img_nums[i]]['white'])
      roi_white.append(roi_set_white)

  for i in range(len(img_nums)):
    if 'amber' in data[img_nums[i]]:
      roi_set_amber = get_roi(image_path + 'frame'+str(img_nums[i])+'.png', data[img_nums[i]]['amber'])
      roi_amber.append(roi_set_amber)
  print("######### AMBER ANALYSIS #########")
  analyze_colorspace_thresh_ColorSpace(roi_amber, 'YCbCr')

  print("######### WHITE ANALYSIS #########")
  analyze_colorspace_thresh_ColorSpace(roi_white, 'YCbCr')
main()