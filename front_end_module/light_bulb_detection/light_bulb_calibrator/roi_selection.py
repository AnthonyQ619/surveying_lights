import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from scipy.stats import norm
from scipy.optimize import curve_fit


def csv_roi_reader(filename, lights=False):
  data = np.loadtxt(filename,
                 delimiter=",", dtype=str)

  frameData = {}
  tempData = data[1:,:]#.astype(dtype = int)
  start_ind = 0
  prevFrame = int(tempData[0,0])

  if lights:
    for row in range(tempData.shape[0]):
      frameData[tempData[row,0].astype(dtype = int)] = {}

    prevColor = tempData[0,1]

    for row in range(tempData.shape[0]):
      currFrame = tempData[row,0].astype(dtype = int)
      currColor = tempData[row,1]
      if currFrame != prevFrame:
        frameData[prevFrame][prevColor] = tempData[start_ind:row,2:].astype(dtype = int)
        prevColor = currColor
        start_ind = row
      elif (row + 1) == tempData.shape[0]:
        frameData[prevFrame][prevColor] = tempData[start_ind:,2:].astype(dtype = int)
      if currColor != prevColor:
        frameData[prevFrame][prevColor] = tempData[start_ind:row,2:].astype(dtype = int)
        start_ind = row
        prevColor = currColor
      prevFrame = currFrame
  else:
    for row in range(tempData.shape[0]):
      currFrame = int(tempData[row,0])
      if currFrame != prevFrame:
        frameData[prevFrame] = tempData[start_ind:row,2:].astype(dtype = int)
        start_ind = row
      elif (row + 1) == tempData.shape[0]:
        frameData[prevFrame] = tempData[start_ind:,2:].astype(dtype = int)
      prevFrame = currFrame

  return frameData

def get_roi(imageName, ROIs):
  imgBGR = cv2.imread(imageName)
  imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

  roi_set = []

  for row in range(ROIs.shape[0]):
    x1, y1, x2, y2 = ROIs[row]
    roi_img = copy.deepcopy(imgRGB)[y1:y2, x1:x2]
    #roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    roi_set.append(roi_img)

  return roi_set

def plot_labeled_roi(imgName, data, frame):
  img = cv2.imread(imgName)
  img = copy.deepcopy(img)

  #Plotting Rectangles
  for i in range(data[frame].shape[0]):
    x1, y1, x2, y2 = data[frame][i]
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

  cv2_imshow(img)

