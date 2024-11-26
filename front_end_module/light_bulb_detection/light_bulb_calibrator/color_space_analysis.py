import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from scipy.stats import norm
from scipy.optimize import curve_fit

# Image Analysis Functions

def analyze_image(image, colorspace):
  f, axarr = plt.subplots(2,3, tight_layout=True, figsize=(8, 8))

  xLabels = [c for c in colorspace]
  for i in range(3):
    #color_img_chn = np.transpose(image[:,:,i], (1, 2, 0))
    row = 0
    col = i
    axarr[row, col].imshow(image[:,:,i])
    axarr[row, col].axis('off')
    axarr[row, col].set_title("Image Channel: " + xLabels[i])
  meanVals = []
  minVals = []
  maxVals = []
  stdVals = []

  for i in range(3):
    #transposed_image = np.transpose(neighbors[i], (1, 2, 0))
    meanVals.append((image[:,:,i]).mean())
    maxVals.append(image[:,:,i].max())
    minVals.append(image[:,:,i].min())
    stdVals.append(image[:,:,i].reshape(1,image[:,:,i].shape[0]*image[:,:,i].shape[1]).std())

  for i in range(3):
    row = 1
    col = i
    if i == 0:
      axarr[row, col].bar(xLabels, meanVals)
      axarr[row, col].set_title("Mean Values of " + colorspace)
    if i == 1:
      axarr[row, col].bar(xLabels, stdVals)
      axarr[row, col].set_title("STD Values of " + colorspace)
    if i ==2:
      axarr[row, col].plot(xLabels, maxVals)
      axarr[row, col].plot(xLabels, minVals)
      axarr[row, col].set_title("Min/Max values of " + colorspace)

def analyze_colorspace_thresh(roi_set, frame):
  f, axarr = plt.subplots(len(roi_set),4, tight_layout=True, figsize=(20,8))

  for i in range(len(roi_set)):
    img = roi_set[i]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    alpha = 0.65
    beta = 0.3
    I_sv = alpha*s + beta*v

    mean, median, std = np.mean(I_sv), np.median(I_sv), np.std(I_sv)
    xLabels = ['Mean', 'Median', 'STD']
    if len(roi_set) > 1:
      axarr[i,1].imshow(img)
      axarr[i,0].imshow(I_sv, cmap='gray')
      axarr[i,0].axis('off')
      axarr[i,1].axis('off')
      axarr[i,0].set_title("Frame Number: " + str(frame))
      axarr[i,1].set_title("Frame Number Colored: "+ str(frame))

      h,w = I_sv .shape[0], I_sv .shape[1]
      img_vec = I_sv .reshape((h*w, 1))
      axarr[i,2].set_title("Frame Number: " + str(frame) + "Pixel Count")
      axarr[i,2].hist(img_vec)
      axarr[i,3].bar(xLabels, [mean, median, std])
    else:
      axarr[1].imshow(img)
      axarr[0].imshow(I_sv, cmap='gray')
      axarr[0].axis('off')
      axarr[1].axis('off')
      axarr[0].set_title("Frame Number: " + str(frame))
      axarr[1].set_title("Frame Number Colored: "+ str(frame))

      h,w = I_sv .shape[0], I_sv .shape[1]
      img_vec = I_sv .reshape((h*w, 1))
      axarr[2].set_title("Frame Number: " + str(frame) + " Pixel Count")
      axarr[2].hist(img_vec)
      axarr[3].bar(xLabels, [mean, median, std])

def analyze_colorspace_thresh_ColorSpace(roi_sets, colorSpace, th = 0.0):
  f, axarr = plt.subplots(2,3, tight_layout=True, figsize=(10,6))
  if colorSpace.lower() == 'hsv':
    channels = ['H', 'S', 'V']
    colorSp = cv2.COLOR_RGB2HSV
  else:
    channels = ["Y", "Cb", "Cr"]
    colorSp = cv2.COLOR_RGB2YCrCb

  h_pixels = np.array([])
  s_pixels = np.array([])
  v_pixels = np.array([])

  for roi_set in roi_sets:
    for i in range(len(roi_set)):
      img = roi_set[i].astype('uint8')
      hsv = cv2.cvtColor(img, colorSp)

      h, s, v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]

      height, width = h.shape[0], h.shape[1]
      imgH_vec = h.reshape((height*width, 1))
      imgS_vec = s.reshape((height*width, 1))
      imgV_vec = v.reshape((height*width, 1))

      h_pixels = np.append(h_pixels, imgH_vec)
      s_pixels = np.append(s_pixels, imgS_vec)
      v_pixels = np.append(v_pixels, imgV_vec)

  axarr[0,0].set_title("Pixel Count of " + channels[0] + " channel")
  axarr[0,0].hist(h_pixels)

  axarr[0,1].set_title("Pixel Count of " + channels[1] + " channel")
  axarr[0,1].hist(s_pixels)

  axarr[0,2].set_title("Pixel Count of " + channels[2] + " channel")
  axarr[0,2].hist(v_pixels)

  if th > 0.0:
    indexC1 = np.argwhere(h_pixels < th)
    indexC2 = np.argwhere(s_pixels < th)
    indexC3 = np.argwhere(v_pixels < th)
    h_pixels = np.delete(h_pixels, indexC1)
    s_pixels = np.delete(s_pixels, indexC2)
    v_pixels = np.delete(v_pixels, indexC3)


  h_mean, s_mean, v_mean = h_pixels.mean(), s_pixels.mean(), v_pixels.mean()
  h_median, s_median, v_median = np.median(h_pixels), np.median(s_pixels), np.median(v_pixels)
  h_std, s_std, v_std = h_pixels.std(), s_pixels.std(), v_pixels.std()

  axarr[1,0].set_title("Mean pixels of all channel")
  axarr[1,0].bar(channels, [h_mean, s_mean, v_mean])
  axarr[1,1].set_title("Median pixels of all channel")
  axarr[1,1].bar(channels,[h_median, s_median, v_median])
  axarr[1,2].set_title("Standard Deviation pixels of all channel")
  axarr[1,2].bar(channels, [h_std, s_std, v_std])

  print("H, S, V (Means):",[h_mean, s_mean, v_mean])
  print("H, S, V (Median):",[h_median, s_median, v_median])
  print("H, S, V (Standard Deviation):",[h_std, s_std, v_std])

  print("Range for " + channels[0] + ":",max(0, h_median - h_std), ",",  h_median + h_std)
  print("Range for " + channels[1] + ":", max(0, s_median - s_std), ",",  s_median + s_std)
  print("Range for " + channels[2] + ":", max(0, v_mean - v_std), ",",  v_mean + v_std)


def func(x, A, beta, B, mu, sigma):
    return (A * np.exp(-x/beta) + B * np.exp(-100.0 * (x - mu)**2 / (2 * sigma**2))) #Normal distribution


def analyze_colorspace_thresh_prob(roi_sets, colorSpace, aboveP = 0.0):
  f, axarr = plt.subplots(3,1, figsize=(5,10))

  if colorSpace.lower() == 'hsv':
    channels = ['H', 'S', 'V']
    colorSp = cv2.COLOR_RGB2HSV
  else:
    channels = ["Y", "Cb", "Cr"]
    colorSp = cv2.COLOR_RGB2YCrCb

  c1_pixels = np.array([])
  c2_pixels = np.array([])
  c3_pixels = np.array([])

  roi_counts = {}
  for roi_set in roi_sets:
    for i in range(len(roi_set)):
      img = roi_set[i].astype('uint8')
      imgCsP = cv2.cvtColor(img, colorSp)
      c1, c2, c3 = imgCsP[:,:,0],imgCsP[:,:,1],imgCsP[:,:,2]


      height, width = c1.shape[0], c1.shape[1]
      imgC1_vec = c1.reshape((height*width, 1))
      imgC2_vec = c2.reshape((height*width, 1))
      imgC3_vec = c3.reshape((height*width, 1))

      c1_pixels = np.append(c1_pixels, imgC1_vec)
      c2_pixels = np.append(c2_pixels, imgC2_vec)
      c3_pixels = np.append(c3_pixels, imgC3_vec)

  c1_count = {}
  c2_count = {}
  c3_count = {}

  for i in range(int(np.max(c1_pixels)) + 1):
    c1_count[i] = 0

  for i in range(int(max(np.max(c2_pixels), np.max(c3_pixels))) + 1):
    c2_count[i] = 0
    c3_count[i] = 0

  for i in range(c1_pixels.shape[0]):
    c1_count[c1_pixels[i]] += 1
    c2_count[c2_pixels[i]] += 1
    c3_count[c3_pixels[i]] += 1

  sumC1 = c1_pixels.shape[0]
  sumC2 = c1_pixels.shape[0]
  sumC3 = c1_pixels.shape[0]

  rpC1_prob = {}
  rpC2_prob = {}
  rpC3_prob = {}

  for i in range(int(np.max(c1_pixels)) + 1):
    rpC1_prob[i] = c1_count[i]/sumC1

  for i in range(int(max(np.max(c2_pixels), np.max(c3_pixels))) + 1):
    rpC2_prob[i] = c2_count[i]/sumC2
    rpC3_prob[i] = c3_count[i]/sumC3

  rpProbs = [rpC1_prob, rpC2_prob, rpC3_prob]
  for i in range(3):
    cVals = []
    cYVals = []

    for key, value in rpProbs[i].items():
      if value >= aboveP:
        cVals.append(key)
        cYVals.append(value)

    axarr[i].set_title("Pixel Value for channel " + channels[i])
    axarr[i].set_ylim([0,0.08])
    axarr[i].bar(cVals, cYVals)

def analyze_colorspace_thresh_HSV_Single(roi_set, frame):
  f, axarr = plt.subplots(len(roi_set)*2,3, tight_layout=True, figsize=(15,15))
  plot_ind = 0
  for i in range(len(roi_set)):
    img = roi_set[i].astype('uint8')
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]

    height, width = h.shape[0], h.shape[1]
    imgH_vec = h.reshape((height*width, 1))
    imgS_vec = s.reshape((height*width, 1))
    imgV_vec = v.reshape((height*width, 1))

    h_mean, s_mean, v_mean = imgH_vec.mean(), imgS_vec.mean(), imgV_vec.mean()
    h_median, s_median, v_median = np.median(imgH_vec), np.median(imgS_vec), np.median(imgV_vec)
    h_std, s_std, v_std = imgH_vec.std(), imgS_vec.std(), imgV_vec.std()



    axarr[plot_ind,0].set_title("Pixel Count of H channel")
    axarr[plot_ind,0].hist(imgH_vec)
    axarr[plot_ind,1].set_title("Pixel Count of S channel")
    axarr[plot_ind,1].hist(imgS_vec)
    axarr[plot_ind,2].set_title("Pixel Count of V channel")
    axarr[plot_ind,2].hist(imgV_vec)


    axarr[plot_ind+1,0].set_title("Mean pixels of all channel")
    axarr[plot_ind+1,0].bar(['H', 'S', 'V'], [h_mean, s_mean, v_mean])
    axarr[plot_ind+1,1].set_title("Median pixels of all channel")
    axarr[plot_ind+1,1].bar(['H', 'S', 'V'],[h_median, s_median, v_median])
    axarr[plot_ind+1,2].set_title("Standard Deviation pixels of all channel")
    axarr[plot_ind+1,2].bar(['H', 'S', 'V'], [h_std, s_std, v_std])

    print("H, S, V (Means):",[h_mean, s_mean, v_mean])
    print("H, S, V (Median):",[h_median, s_median, v_median])
    print("H, S, V (Standard Deviation):",[h_std, s_std, v_std])

    print("Range for H:",max(0, h_median - h_std), ",",  h_median + h_std)
    print("Range for S:", max(0, s_median - s_std), ",",  s_median + s_std)
    print("Range for V:", max(0, v_mean - v_std), ",",  v_mean + v_std)

    plot_ind += 2
    cv2.imshow(cv2.cvtColor(cv2.resize(img, (height*4, width*4)), cv2.COLOR_RGB2BGR))
    cv2.waitKey()