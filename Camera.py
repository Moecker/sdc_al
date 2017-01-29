import numpy as np
import cv2
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimage

from Calibration import Calibration
       
class Camera:
    def __init__(self, corners_x, corners_y, calibration=Calibration(), is_debug=False):
        # Base points for transformation
        kBaseLeft = 200
        kBaseRight = 900
        kBaseBottom = 720
        # Number of corners in x-direction for calibration images
        self.corners_x = corners_x
        # Number of corners in y-direction for calibration images
        self.corners_y = corners_y
        # Source points for perspective transformation
        self.source_points = np.float32([[200,kBaseBottom],
                                         [594,450],
                                         [688,450],
                                         [1100,kBaseBottom]])
        # Destination points for perspective transformation
        self.destination_points = np.float32([[kBaseLeft,kBaseBottom],
                                              [kBaseLeft,0],
                                              [kBaseRight,0],
                                              [kBaseRight,kBaseBottom]])
        # Camera calibraton
        self.calibration = calibration
        # Debug mode
        self.is_debug = is_debug
        # Perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        # Inverse Perspective transformation matrix
        self.inverse_M = cv2.getPerspectiveTransform(self.destination_points, self.source_points)

    def calibrate(self, file_list):
        print("Calibrating camera ...")

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        target_object_points = np.zeros((self.corners_x * self.corners_y, 3), np.float32)
        target_object_points[:, :2] = np.mgrid[0:self.corners_x, 0:self.corners_y].T.reshape(-1, 2)

        object_points = [] # 3D points in real world space
        image_points = [] # 2S points in image plane

        images = glob.glob(file_list)
        
        # Read in images and find chessboard corners
        for fname in images:
            # Read in 
            image = mpimage.imread(fname)
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.corners_x, self.corners_y), None)

            # If corners are found, add object points, image points
            if ret == True:
                image_points.append(corners)
                object_points.append(target_object_points)

                # draw and display the corners
                image = cv2.drawChessboardCorners(image, (self.corners_x, self.corners_y), corners, ret)
            else:
                print("Warning: Correct number of corners for image {}".format(fname) + " was not found")

        # Get camera calibration params
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, 
                                                           image_points, 
                                                           gray.shape[::-1], 
                                                           None, 
                                                           None)
        self.calibration = Calibration(mtx, dist)
       
        
    def undistort_image(self, image):
        print("Undistorting image ...")
            
        # Apply distortion correction to the image
        undistorted_image = cv2.undistort(image, 
                                          self.calibration.mtx, 
                                          self.calibration.dist, 
                                          None, 
                                          self.calibration.mtx)
        return undistorted_image

        
    def perspective_transformation(self, image):
        print("Perspecive transformation ...")
        
        # Apply he perspective transformation
        image_size = (image.shape[1], image.shape[0])
        
        transformed_image = cv2.warpPerspective(image, 
                                                self.M, 
                                                image_size, 
                                                flags=cv2.INTER_LINEAR)
        return transformed_image
       
          
    def detect_edges(self, image):
        print("Detecting edges ...")
       
        # 1. Sobel filtering
        # Grayscale the image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_tresholded = self._apply_sobel_(gray)
       
        # 2. Color tresholding
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        saturation = hls[:,:,2]
        color_tresholded = self._apply_color_(saturation)
     
        # 3. Combination of sobel and color thresholding
        # Combine the two binary thresholds
        combined_tresholded = np.zeros_like(sobel_tresholded)
        combined_tresholded[(color_tresholded == 1) | (sobel_tresholded == 1)] = 1

        return combined_tresholded, sobel_tresholded, color_tresholded
        
        
    def _apply_sobel_(self, gray):
        kGradientMin = 20
        kGradientMax = 200
        
        # Take the derivative only in x
        sobel_x_filter = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        
         # Absolute x derivative to accentuate lines away from horizontal
        sobel_x_filter_abs = np.absolute(sobel_x_filter)
        
        # Scale the filter according to max value
        scaled_sobel = np.uint8(255 * sobel_x_filter_abs / np.max(sobel_x_filter_abs))

        # Threshold x gradient
        sobel_tresholded = np.zeros_like(scaled_sobel)

        ret, threshold = cv2.threshold(scaled_sobel, kGradientMin, kGradientMax, cv2.THRESH_BINARY)
        sobel_tresholded[(threshold >= kGradientMin) & (threshold <= kGradientMax)] = 1
        
        return sobel_tresholded
        
        
    def _apply_color_(self, saturation):
        # Threshold color channel
        kSaturationMin = 170
        kSaturationMax = 240
        
        color_tresholded = np.zeros_like(saturation)
        threshold = cv2.inRange(saturation.astype('uint8'), kSaturationMin, kSaturationMax)

        color_tresholded[(threshold == 255)] = 1
        
        return color_tresholded
