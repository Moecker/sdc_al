import numpy as np
import glob
import cv2
import pickle

import matplotlib.pyplot as plt

kCornersX = 9 # the number of inside corners in x
kCornersY = 6 # the number of inside corners in y
kCalibrationFolder = 'camera_cal/'
kCalibrationOutputFolder = 'camera_cal_output/'

def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((kCornersX * kCornersY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:kCornersX, 0:kCornersY].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(kCalibrationFolder + 'calibration*.jpg')
    images = sorted(images)
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print("Using " + fname)
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (kCornersX, kCornersY), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (kCornersX, kCornersY), corners, ret)
            
            write_name = kCalibrationOutputFolder + 'corners' + str(idx) + '.jpg'
            cv2.imwrite(write_name, img)
            
    # Test undistortion on an image
    img = cv2.imread(kCalibrationOutputFolder + 'test_calibration.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(kCalibrationOutputFolder + 'test_calibration_undistorted.jpg', dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open('calibration.p', 'wb'))
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()