import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal, stats
from scipy.signal import find_peaks
import cv2

# Helper Function
def read_video_cv2(file_name, n_frames=80):
    cap = cv2.VideoCapture(file_name)
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        arr = np.array(gray_frame)
        all.append(arr)
        i += 1
    return np.array(all)

def center_of_intensity_threshold(image, threshold=0):
    # Threshold the image to focus on specific intensity ranges
    thresholded_image = image.copy()
    thresholded_image[thresholded_image < threshold] = 0

    # Creating grid of x and y coordinates
    y, x = np.indices(thresholded_image.shape)

    # Computing total intensity
    total_intensity = np.sum(thresholded_image)
    # print(total_intensity)


    # Computing center of intensity (weighted average)
    center_x = np.sum(x * thresholded_image) / total_intensity
    center_y = np.sum(y * thresholded_image) / total_intensity

    center_x = math.floor(center_x )
    center_y = math.floor(center_y )


    return center_x, center_y



def compute_mean_intensity_cropped_image(data):
    # Create an empty array to store mean intensity for each frame
    mean_intensity_per_frame = np.zeros(np.shape(data)[0], dtype=float)
    # print(np.shape(mean_intensity_per_frame))

    # Iterate over each frame
    for i in range(data.shape[0]):
        # Select pixels less than 255 and greater than 0 for the current frame
        # valid_pixels = data[i][(data[i] < 1) & (data[i] > 0)]
        # valid_pixels = data[i][(data[i] < 1)]

        valid_pixels = data[i][(data[i] > 0)]

        # Compute mean intensity for the current frame
        mean_intensity_per_frame[i] = np.mean(valid_pixels, axis = 0)


    return mean_intensity_per_frame


def compute_mean_intensity_circular_region(data, center_x, center_y, radius):
  # Generate a grid of coordinates within the image
  x, y = np.meshgrid(np.arange(data.shape[2]), np.arange(data.shape[1]))

  # Calculate the distance from each point to the center
  distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
  # pixel_select = data[(data<254)]

  # Create a circular mask where True represents pixels within the circle
  circular_mask = (distance <= radius)


  # Apply the circular mask to the data to select pixels within the circle
  pixels_within_circle = data[:, circular_mask]
  # print(np.shape(pixels_within_circle))


  # # Calculate the mean intensity within the circular region
  mean_intensity = np.mean(pixels_within_circle, axis=1)
  peak_intensity = np.max(pixels_within_circle)

  # circular_mask = pixel_select

  return mean_intensity, peak_intensity, circular_mask

def compute_raw_brf_signal(vid_data, topL_pt, botR_pt, px_radius, x_offset, y_offset, extraction_methods, anlys_pts):

  cols = len(extraction_methods)
  rows = 1

  fig, axs = plt.subplots(rows, cols, figsize=(25, 4))
  brf_signals = []
  for method in range(len(extraction_methods)):

    y0, y1 = topL_pt[1],botR_pt[1]
    x0, x1 = topL_pt[0],botR_pt[0]
    data0 = vid_data[:, y0:y1, x0:x1]

    px_range = px_radius
    x_cm, y_cm = anlys_pts[0]+x_offset, anlys_pts[1] + y_offset #20, int(rad)

    sig_cm = data0[:, y_cm+px_range,x_cm+px_range]

    # sig_mean = np.mean(data0[:, y_cm-px_range:y_cm+px_range, x_cm-px_range:x_cm+px_range],axis=(1,2))


    radius = px_range
    mean_intensity_circular_region, peak_intensity_circular_region, circular_mask  = compute_mean_intensity_circular_region(data0, x_cm, y_cm, radius)

    mean_intensity_cropped = compute_mean_intensity_cropped_image(data0)


    def plot_brf(brf,ax,peaks):
      if peaks:
        peak_indices = signal.find_peaks(brf, distance = 1.5)[0]
        nadir_indices = signal.find_peaks(-brf, distance = 1.5)[0]
        ax.plot(brf[peak_indices[0]:nadir_indices[-1]])
        ax.set_title(f"{extraction_methods[method]}_Raw Bulb characteristic")
        # ax.show
      else:
        ax.plot(brf)
        ax.set_title(f"{extraction_methods[method]}_Raw Bulb characteristic")
        # plt.show()

    if  extraction_methods[method] == "Pruned pixels": #pruning the saturated pixels
      brf_pp = mean_intensity_cropped
      plot_brf(brf_pp, axs[1], False)
      brf_signals.append(brf_pp)
    elif  extraction_methods[method] == "Circular region": #selecting the entire bulb blob
      brf_cr = mean_intensity_circular_region
      plot_brf(brf_cr,axs[2],  False)
      brf_signals.append(brf_cr)
    else:
      brf_scm = sig_cm
      plot_brf(brf_scm,axs[0], False)
      brf_signals.append(brf_scm)

  fig = plt.figure(figsize=(10,5))
  ax = fig.add_subplot(111)

  ax.imshow(data0[6],cmap='gray')

  ax.set_title('points')
  points = ax.scatter(x_cm,y_cm,c='r')

  plt.show()

  return brf_signals