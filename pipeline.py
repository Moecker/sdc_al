# Imports
import numpy as np
import cv2
import pickle

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Imports from modules
import warping as warp

# Constants
kOutputFolder = 'output_images/'
kTestImagesFolder = 'test_images/'

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open('calibration.p', 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread(kTestImagesFolder + 'straight_lines1.jpg')
img_size = img.shape[0:2][::-1]

# Pipeline
# 1. Undistort
undist = cv2.undistort(img, mtx, dist, None, mtx)

# 2. Warp
warped = warp.warp_street(undist)

fig = plt.figure()

x = [warp.bottom_left[0], warp.top_left_src[0], warp.top_right_src[0], warp.bottom_right[0]]
y = [warp.bottom_left[1], warp.top_left_src[1], warp.top_right_src[1], warp.bottom_right[1]]

plt.subplot(211)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.plot(x, y, 'r-', lw=1)
plt.xlim([0, img_size[0]])
plt.ylim([img_size[1], 0])

x = [warp.bottom_left[0], warp.top_left[0], warp.top_right[0], warp.bottom_right[0]]
y = [warp.bottom_left[1], warp.top_left[1], warp.top_right[1], warp.bottom_right[1]]

plt.subplot(212)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.plot(x, y, 'r-', lw=1)
plt.xlim([0, img_size[0]])
plt.ylim([img_size[1], 0])

plt.show()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.savefig("warped.png")

cv2.imwrite(kOutputFolder + 'straight_lines1_warped.jpg', warped)