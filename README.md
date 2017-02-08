## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_src]: ./output_images/calibration/test_calibration.jpg "Test Calibration Image"
[cal_src_chess]: ./output_images/calibration/test_calibration_chessboard_corners.jpg "Test Calibration Image Chessboard Corners"
[cal_src_undist]: ./output_images/calibration/test_calibration_undistorted.jpg "Test Calibration Image Undistorted"
[cal_src_undist_warped]: ./output_images/calibration/test_calibration_undistorted_warped.jpg "Test Calibration Image Undistorted Warped"

[undist1]: ./output_images/pipeline_readme/undistorted_plot_1.png
[undist2]: ./output_images/pipeline_readme/undistorted_plot.png

[thresh1]: ./output_images/pipeline_readme/thresholds_plot_1.png
[thresh2]: ./output_images/pipeline_readme/thresholds_plot.png

[bird1]: ./output_images/pipeline_readme/birdeye_plot_1.png
[bird2]: ./output_images/pipeline_readme/birdeye_plot.png

[binbird]: ./output_images/pipeline_readme/binary_birdeye_plot.png
[bin]: ./output_images/pipeline_readme/binary_plot.png

[lextr]: ./output_images/pipeline_readme/left_extracted_plot.png
[rextr]: ./output_images/pipeline_readme/right_extracted_plot.png

[lfit]: ./output_images/pipeline_readme/left_fitted.png
[rfit]: ./output_images/pipeline_readme/right_fitted.png
[fit]: ./output_images/pipeline_readme/both_fitted.png

[combbird]: ./output_images/pipeline_readme/combined_birdeye_plot.png
[comb]: ./output_images/pipeline_readme/combined_plot.png

[pers]: ./output_images/pipeline/perpective.png
[log]: ./output_images/logging.png

[video1]: ./output_videos/project_video_processed.mp4 "Project Video"

## Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 
The full Rubrics definition is stated on the Udacity page: https://review.udacity.com/#!/rubrics/571/view

---

###General Remarks / Software Structure

As this project was failry involved, a nice and debug & logging frindly software structure was essential. The project is principally modularized into these components:

Classes
* [Camera](Camera.py): Handles all camera calibrating and undistortion and warping.
* [Calibration](Calibration.py): Simple storage for camera distortion coefficients.
* [Lane](Lane.py): Contains all Lane related code, such as the histogram search for both lines and the general tracking of those.
* [Line](Line.py): Implements mainly the fitting and evaluation methods as well as the smoothing over time.

Runners
* [main](main.py)
* [plotting](plotting.py)
* [udacity_code](udacity_code.py)

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

May I cite you guys here: "You're reading it!" :)

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All camera related code is implemented inside the [Camera](Camera.py) class. It is constructed by passing these parameters:
* Number of corners of the chessboard images used for calibration: `corners_x`, `corners_y`, 
* Activation of Debugging and Logging functionality: `log`, `is_debug`, `mode`,
* Previously calibrated camera matrices to skip calibration: `calibration`

Note: This section is basically adopted from the writeup template as there is no essential difference or addition to mention.
I start by preparing object points (named `source_points` in code, as this is the source for the latter undistortion), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `source_points` is just a replicated array of coordinates, and it will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  The image points (named `destination_points` in code, as this is the destination for the latter undistortion) will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

The chessboard corners have been detected using the `cv2.findChessboardCorners` method applied on a previously grayscaled image, handing over the expected number of chessboard corners in x and y. The `cv2.drawChessboardCorners` method eases drawing the detected corners.

I then used the output source_points and destination_points to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained following results.

|Source Test Image|Detected Chessboard Corners| 
|---|---|
|![alt text][cal_src]|![alt text][cal_src_chess]|

|Undistored Image|Undistorted & Warped Image| 
|---|---|
|![alt text][cal_src_undist]|![alt text][cal_src_undist_warped]|

---

###Pipeline (Single Images)

####0. Pipeline Overview

The pipeline which was applied for images and videos, was established as follows. Preconditions impose the already obtained distortion matrix during the calibration step. To better cache the implemntation in code, I added a log output screenshot revealing each step. For the video pipeline modifications of keeping trakc of previous fits applied, as described in the latter rubrics. The pipeline is triggered in the [main](main.py) script at line #111 in the ´process_image method´.

1. Undistort the image
2. Apply perpective transform
3. Detect edges using thresholds (binary)
4. Locate lane lines on the binary
5. Fit both lines
6. Check fit validity
7. Draw lane onto road

![alt text][log]

####1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like the following one. The left image shows the undistorted, source image - the right one shows the same image applied with the disortion Matrix obtained during the camera calibration step. Although no fundamental difference is observed, the undistorted image gives a more "true" representation of the actual lane's shape.

|Distorted Image|Undistorted Image| 
|---|---|
|![alt text][undist1]|![alt text][undist2]|

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Those thersholds are applied in the [Camera](Camera.py) class in line #132 in method `detect_edges`. First, both thresholds are applied independently: 
* The color thresholds using the saturation (S) channel of a previously HLS color space converted images. This is implemented in line #176 in the `_apply_color_` method
* The gradient threshold using basically a Sobel filter in x-direction applied on a previously grayscaled image. The actual method can be found in line #154 in method `_apply_sobel_`.

The thresholds are parameterized and those combintation revealed the optimal results for the project video.
```
kSaturationMin = 175 # The minimum saturation of the pixel to be considered.
kSaturationMax = 240 # Likewise the maximum saturation.
kGradientMin = 15    # The minimum gradient (simple: the amount of change in grayscale value within an area of pixels)
kGradientMax = 100   # Likewise the maximum allowed gradient.
```

Secondly, both thresholds were combined by a simple or-concatination revealing a sensitive treshhold by exploiting both methods advantages. For instance, the color threshhold was more robust against brightness changes, wheras the gradient revealed in general more exact results in masking the lines.

Both thresholds are neither absolute max or min values, but were rather obtained experimentally and resulted in reasonable findings as one can see in the [output_images/pipeline](output_images/pipeline) folder.

Here's an example of my output for this step. The most left images displays the result of the gradient treshold on the perspective-transformed (warped) image, the center image the result of the color (or saturation) threshold, and the right shows the combined treshhold applied on the original image.

|Threshold Gradient|Threshold Color|Both Thresholds Applied| 
|---|---|---|
|![alt text][thresh1]|![alt text][thresh2]|![alt text][bin]|

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the class [Camera.py](Camera) in line #119 in the method `perspective_transformation`. It basically applies the open-cv method ´cv2.warpPerspective´ on the input image, using the previously obtained perspective transformation matrix `M`. This matrix (and also the `inverse_M` matrix, which is latter required to un-warp the resulting lane visualization back onto the road) are computed only once during instanciation of the `Camera` class using the `cv2.getPerspectiveTransform` method. 

This perspective transform method takes as input four source and four destination points, whereby each of the source points is mapped to the designated destination point. The points were chossen as such that for a straight line image the lines on the prepective transormed image remained straight. Another requirement for the transform was that there was a wide enough area left and right of the lines for anticipating images with strong curves.

I chose to hardcode the source and destination points in the following manner:

```
# Source points for perspective transformation
self.source_points = np.float32([[200,kBaseBottom],
                                 [594,kBaseCenter],
                                 [688,kBaseCenter],
                                 [1100,kBaseBottom]])
                                 
# Destination points for perspective transformation
self.destination_points = np.float32([[kBaseLeft,kBaseBottom],
                                      [kBaseLeft,0],
                                      [kBaseRight,0],
                                      [kBaseRight,kBaseBottom]])

```
with 
```       
kBaseLeft = 200
kBaseRight = 900
kBaseBottom = 720
kBaseCenter = 450
```  

This resulted in the following source and destination points:

| Source      | Destination  | 
|:-----------:|:------------:| 
| (200,720)   | (200, 20)    | 
| (594, 450)  | (200, 0)     |
| (688, 450)  | (900, 0)     |
| (1100,720)  | (900,720)    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][pers]

|Input Image|Perpective Transform: Birdeye View| 
|---|---|
|![alt text][bird2]|![alt text][bird1]|

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

|Left Extracted|Right Extracted| 
|---|---|
|![alt text][lextr]|![alt text][rextr]|

|Left Fitted|Right Fitted| 
|---|---|
|![alt text][lfit]|![alt text][rfit]|

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

|Combined Birdeye|Combined Resulting Image| 
|---|---|
|![alt text][combbird]|![alt text][comb]|

---

###Pipeline (Video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

