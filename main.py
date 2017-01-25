# Imports
import numpy as np
import cv2
import pickle

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Imports from modules
import camera_calibration as calibration
import warping as warping

# Constants
kCornersX = 9 # the number of inside corners in x
kCornersY = 6 # the number of inside corners in y
kCalibrationOutputFolder = 'camera_cal_output/'

# Calibration
calibration.calibrate()

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open('calibration.p', 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread(kCalibrationOutputFolder + 'test_calibration.jpg')

top_down, perspective_M = unwarp.corners_unwarp(img, kCornersX, kCornersY, mtx, dist)
cv2.imwrite(kCalibrationOutputFolder + 'test_calibration_undistorted_warped.jpg', top_down)

# Plotting
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()