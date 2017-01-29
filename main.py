import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import logging as log
import sys
log.basicConfig(stream=sys.stderr, level=log.WARN)

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
is_lane_debug = False
shall_calibrate = False;
plot_output = False

camera = None
lane = None
    

def main(use_images=False, use_video=False):
    log.info("# 0. Prepare system")
    # Load a previously conducted calibration
    if not shall_calibrate:
        calibration = load_camera_calibration(kCalibrationPickleFile)
    else:
        calibration = None

    # Create a camera object
    global camera
    camera = Camera(kCornersX, kCornersY, log=log, calibration=calibration, is_debug=is_camera_debug)
    
    # Create a lane object
    global lane
    lane = Lane(log=log, is_debug=is_lane_debug)

    # Calibration
    if shall_calibrate:
        # Calibrate the camera
        camera.calibrate(kCalibrationFolder + 'calibration*.jpg')
        # Save the calibration
        save_calibration(camera, kCalibrationPickleFile)
    
    if use_images: run_images(camera, lane)
    if use_video: run_video(camera, lane)
    if not use_images and not use_video: print("Warning: No mode selected")


def run_video(camera, lane):
    log.info("Running video ...")

    from moviepy.editor import VideoFileClip

    clip = VideoFileClip("./project_video.mp4")
    output_video = "./project_video_processed.mp4"

    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)


def run_images(camera, lane):
    log.info("Running images ...")
    images = glob.glob(kTestImagesFolder + 'test*.jpg')
    for file_name in images:
    
        log.info("Processing " + file_name + " ...")
        
        # Read in an image
        image = cv2.imread(file_name)
        frame_name = os.path.basename(file_name).split('.')[0]
              
        # Trigger the processing chain
        process_image(image, frame_name)

    
def process_image(image, frame_name=""):
    # Pipeline
    log.info("# 1. Undistort the image")
    undistorted_image = camera.undistort_image(image)
    if is_debug: plotted = plot.plot_images(image, undistorted_image, frame_name + "_undistorted")

    log.info("# 2. Apply perspective transform")
    birdeye_image = camera.perspective_transformation(undistorted_image)
    if is_debug: plotted = plot.plot_images(undistorted_image, birdeye_image, frame_name + "_birdeye")

    log.info("# 3.1 Detect edges using thresholds in color and canny")
    binary_image, ignore, ignore = camera.detect_edges(undistorted_image)
    if is_debug: plotted = plot.plot_images(undistorted_image, binary_image, frame_name + "_binary", is_gray=True)
    
    log.info("# 3.2 Detect edges on the warped image")
    binary_birdeye, sobel, color = camera.detect_edges(birdeye_image)
    if is_debug: plotted = plot.plot_images(birdeye_image, binary_birdeye, frame_name + "_binary_birdeye", is_gray=True)
    if is_debug: plotted = plot.plot_images(sobel, color, frame_name + "_thresholds", is_gray=True)
    
    log.info("# 4. Locate the lane lines based on the binary image")
    lane.locate_lines(binary_birdeye)
    
    log.info("# 5. Fit the lines of the lane")
    lane.fit_lines()
    
    log.info("# 6. Draw lane back onto the road")
    combined_image, combined_birdeye = lane.draw_lines(undistorted_image, 
        binary_birdeye, birdeye_image, camera.inverse_M)

    if is_debug or plot_output: plotted = plot.plot_images(birdeye_image, combined_birdeye, frame_name + "_combined_birdeye")
    if is_debug or plot_output: plotted = plot.plot_images(undistorted_image, combined_image, frame_name + "_combined")

    return combined_image
        
    
def load_camera_calibration(pickle_file):
    log.info("Loading calibration data ...")
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open(pickle_file, 'rb'))
    return dist_pickle["calibration"]
    
    
def save_calibration(camera, pickle_file):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    log.info("Saving calibration data ...")
    dist_pickle = {}
    dist_pickle["calibration"] = camera.calibration
    pickle.dump(dist_pickle, open(pickle_file, 'wb'))


# Call the main routine
main(use_images=False)
main(use_video=True)