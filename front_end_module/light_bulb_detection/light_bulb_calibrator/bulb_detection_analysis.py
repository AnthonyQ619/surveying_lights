import numpy as np
import matplotlib.pyplot as plt
import cv2
from bulb_detection_analysis import *
from roi_selection import *

def main():
  csv_file = ""

  # Read Data
  data = csv_roi_reader(csv_file, True)
  print(data)