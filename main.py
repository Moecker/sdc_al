# Imports
import numpy as np
import cv2
import pickle
import glob
import os

import plotting as plot

from Camera import Camera
from Lane import Lane

# Constants
kCornersX = 9 # the number of inside corners in x
kCornersY = 6 # the number of inside corners in y

kTestImagesFolder = 'test_images/'
kCalibrationFolder = 'camera_cal/'
kCalibrationPickleFile = 'calibration.p'

is_debug = False
is_camera_debug = False
is_lane_debug = True
shall_calibrate = False;

    
def main():
    # Load a previously conducted calibration
    if not shall_calibrate:
        calibration = load_camera_calibration(kCalibrationPickleFile)
    else:
        calibration = None

    # Create a camera object
    camera = Camera(kCornersX, kCornersY, calibration, is_debug=is_camera_debug)
    
    # Create a lane object
    lane = Lane(is_debug=is_lane_debug)

    # Calibration
    if shall_calibrate:
        # Calibrate the camera
        camera.calibrate(kCalibrationFolder + 'calibration*.jpg')
        # Save the calibration
        save_calibration(camera, kCalibrationPickleFile)
    
    images = glob.glob(kTestImagesFolder + '*test1.jpg')
    
    for file_name in images:
        print("Processing " + file_name + " ...")
        
        # Read in an image
        image = cv2.imread(file_name)
        frame_name = os.path.basename(file_name).split('.')[0]
              
        # Trigger the processing chain
        process_image(image, camera, lane, frame_name)

    
def process_image(image, camera, lane, frame_name=""):
    # Pipeline
    # 1. Undistort the image
    undistorted_image = camera.undistort_image(image)
    if is_debug: plt = plot.plot_images(image, undistorted_image, frame_name + "_undistored")

    # 2. Apply perspective transform 
    transformed_image = camera.perspective_transformation(undistorted_image)
    if is_debug: plt = plot.plot_images(undistorted_image, transformed_image, frame_name + "_transformed")

    # 3.1 Detect edges using thresholds in color and canny
    edge_detected, ignore, ignore = camera.detect_edges(undistorted_image)
    if is_debug: plt = plot.plot_images(undistorted_image, edge_detected, frame_name + "_edge_detected", is_gray=True)
    
    # 3.2 Detect edges on the warped image
    edge_transformed, sobel, color = camera.detect_edges(transformed_image)
    if is_debug: plt = plot.plot_images(transformed_image, edge_transformed, frame_name + "_edge_transformed", is_gray=True)
    if is_debug: plt = plot.plot_images(sobel, color, frame_name + "_sobel_color", is_gray=True)
    
    # 4. Locate the lane lines based on the binary image
    lane.locate_lines(edge_transformed)
    
    # 5. Fit the lines of the lane
    lane.fit_lines()
    
    # 6. Draw lane back onto the road
    resulting_image = lane.draw_lines(undistorted_image, edge_transformed, camera.inverse_M)
    plt = plot.plot_images(undistorted_image, resulting_image, frame_name + "_result")
        
    
def load_camera_calibration(pickle_file):
    print("Loading calibration data ...")
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open(pickle_file, 'rb'))
    return dist_pickle["calibration"]
    
    
def save_calibration(camera, pickle_file):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    print("Saving calibration data ...")
    dist_pickle = {}
    dist_pickle["calibration"] = camera.calibration
    pickle.dump(dist_pickle, open(pickle_file, 'wb'))


# Call the main routine
main()
