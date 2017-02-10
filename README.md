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

[frame]: ./output_images/pipeline_readme/frame_based.png
[histo]: ./output_images/pipeline_readme/histo_based.png

[pers]: ./output_images/pipeline_readme/perpective.png
[log]: ./output_images/logging.png

[video1]: ./output_videos/project_video_processed.mp4 "Project Video"

## Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 
The full Rubrics definition is stated on the Udacity page: https://review.udacity.com/#!/rubrics/571/view

---

###General Remarks / Software Structure

As this project was fairly involved, a nice and debug & logging friendly software structure was essential. The project is principally modularized into these components:

Classes
* [Camera](Camera.py): Handles all camera calibration, undistortion and prespective tranformation.
* [Calibration](Calibration.py): Simple storage for camera distortion coefficients.
* [Lane](Lane.py): Contains all lane related code, such as the histogram search for both lines and the general tracking of those.
* [Line](Line.py): Implements mainly the fitting and evaluation methods as well as the smoothing over time.

Runners
* [main](main.py): The main runner and entry point for the image and video pipeline.
* [plotting](plotting.py): Holds debug code for plotting images of each stage of the pipeline.
* [udacity_code](udacity_code.py): Samples and from udacity suggested code for testing and as an implementation reference.

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

May I cite you here: "You're reading it!" :)

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All camera related code is implemented inside the [Camera](Camera.py) class. It is constructed by passing these parameters:
* Number of corners of the chessboard images used for calibration: `corners_x`, `corners_y`, 
* Activation of Debugging and Logging functionality: `log`, `is_debug`, `mode`,
* Previously calibrated camera matrices to skip calibration: `calibration`

Note: This section is basically adopted from the writeup template as there is no essential difference or addition to mention.
I start by preparing object points (named `source_points` in code, as this is the source for the latter undistortion), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `source_points` is just a replicated array of coordinates, and it will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  The image points (named `destination_points` in code, as this is the destination for the latter undistortion) will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

The chessboard corners have been detected using the `cv2.findChessboardCorners` method applied on a previously grayscaled image, handing over the expected number of chessboard corners in x and y. The `cv2.drawChessboardCorners` method eases drawing the detected corners. I then used the output source_points and destination_points to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained following results.

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
* The color threshold uses the saturation (S) channel of a previously into HLS color space converted image. This is implemented in line #176 in the private `_apply_color_` method. A color threshold assumes that lines have a stronger saturation channel than other environment entities.
* The gradient threshold approach basically incoporates a Sobel filter in x-direction applied on a previously grayscaled image. The actual method can be found in line #154 in method `_apply_sobel_`. The idea behind the Sobel filter is that lane-lines differ substancially from the surrounding street and hence there exist a gradient from street to line marking.

The thresholds are parameterized with min and max values; following combinatation revealed the optimal results for the project video.
```
kSaturationMin = 175 # The minimum saturation of the pixel to be considered.
kSaturationMax = 240 # Likewise the maximum saturation.
kGradientMin = 15    # The minimum gradient (simple: the amount of change in grayscale value within an area of pixels)
kGradientMax = 100   # Likewise the maximum allowed gradient.
```

Secondly, both thresholds were combined by a simple or-concatination, revealing a sensitive threshhold by exploiting both methods advantages. For instance, the color threshhold was more robust against brightness changes, wheras the gradient revealed in general more exact results in masking the lines.

Both thresholds are neither absolute max or min values, but were rather obtained experimentally and resulted in reasonable findings as one can see in the [output_images/pipeline](output_images/pipeline) folder.

Here's an example of my output for this step. The most left image displays the result of the gradient treshold on the perspective-transformed (warped) image, the center image reveales the result of the color (or saturation) threshold, and the most right one shows the combined treshholds applied on the original image.

|Threshold Gradient|Threshold Color|Both Thresholds Applied| 
|---|---|---|
|![alt text][thresh1]|![alt text][thresh2]|![alt text][bin]|

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the class [Camera](Camera.py) in line #119 in the method `perspective_transformation`. It basically applies the open-cv method ´cv2.warpPerspective´ on the input image, using the previously obtained perspective transformation matrix `M`. This matrix (and also the `inverse_M` matrix, which is latter required to un-warp the resulting lane visualization back onto the road) are computed only once during instanciation of the `Camera` class using the `cv2.getPerspectiveTransform` method. 

This perspective transform method takes as input four source and four destination points, whereby each of the source points is mapped to the designated destination point. The points were chossen as such that for a straight line image the lines on the prepective transormed image remained straight. Another requirement for the transform was that there was a wide enough area left and right of the lines anticipating images with strong curves.

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

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][pers]

|Input Image|Perpective Transform: Birdeye View| 
|---|---|
|![alt text][bird2]|![alt text][bird1]|

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The main method for detecting lane-line pixels is implemented in the [Lane](Lane.py) class in method `locate_lines` in line #81. There are two modes of detecting line-points from an image I have implemented: One is the histogram based approached and the other is the frame based approach. Latters is only usable once a line has already been detected in a previous step and hence only applicable in the video pipeline mode.

In the histogram approach, a histogram of the amount of extracted line-points in x-direction is computed. The asusmption behind this approach is that the more line pixels have been extracted at a certain position the more certain this area is part of the actual line. This is rather robust against outliers as outliers usually don't come in clusters. Only line-pixel clusters would contribute to a high histogram value. 

The algorithm is implemented in the `_locate_lines_histo_based_` method in the same class. It first takes the bottom half of the picture for an initial guess of the basis of both lines. The advantage of a initial broad histogram area is that this area's line pixels are mostly best extracted and to enhance robustness. It then uses a moving window approach to detect, based on the initial guess, the next upper search window and so on. In the final version I use `self.kNumberOfSlidingWindows = 8` as a reasonable number of windows. The size of the windows comes automatically with the image's height. Each window supersedes the previous one.

An additional approach is the frame based one implemented also in the [Lane](Lane.py) class in method `_locate_lines_frame_based_` in line #123. This approach can only be applied once a stable fit and hence valid coefficients for the polynomials have been found. It is way less computation expensive than the histogram approach and should always be favored in situations where a stable lane is available. The histogram approach has its advantages in particular when no initial guess about the lines location is available (i.e. at the very beginning), or when a very bad state of polynomial fitting has been detected by the pipeline (which turned out to be not that easy)

The result of the histogram approach with its moving windows is a chain of rectangles as visualized in the left window. The right window shows the frame based approach with a rather continious search area - both marked in green.

|Histogram Approach|Frame Approach| 
|---|---|
|![alt text][histo]|![alt text][frame]|

The final result of extracted line points for left and right line are displayed here. After the area of points has been setup, the actual points are extracted using `self._extract_fit_points_(left_extracted, <the current line>)` in line #118 and #119. Hereby all non-null pixels are used as we already have a binary image.

|Left Extracted|Right Extracted| 
|---|---|
|![alt text][lextr]|![alt text][rextr]|

Based on this extracted line points a polynom of 2nd order is fitted, taken all points into account. To do so, the numpy method `np.polyfit(self.all_y_pixels, self.all_x_pixels, 2)` is used. The number 2 denotes the order. The fit is implemented in the [Line](Line.py) class in the `fit` method in line #60. the resulting (three) coefficients are stored inside the member variable `self.current_coefficients`. Most of the specifica of a Line is further implemented in this class. The Lane class simply instatiates two instances of a Line for right and left.

|Left Fitted|Right Fitted| 
|---|---|
|![alt text][lfit]|![alt text][rfit]|

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The formula to calculate the radius of the curvature is implemented as such: 
```
self.radius_of_curvature = \
    ((1 + (2 * curvature_coeffs[0] * y_eval * ym_per_pix + curvature_coeffs[1]) ** 2) ** 1.5) \
    / np.absolute(2 * curvature_coeffs[0])
```
and called in the [Line](Line.py) class in method `_compute_curvature_` in line #123.

It is important to mention, that the `curvature_coeffs` are not equal to the coefficient computed in the pixel-space fit. This is due to the fact that we cannot simply convert fitting coefficients from pixel to metric space, but rather must reinitiate the fit. This is done in the same method with adapted x and y values of each pixel. The values taken were empirically chosen; the value 3.7m per 700px is an assumption that according to my research the US lanes are typically 3.7m wide. 700px is simply the in-image measured width of the lane.

```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is the last step of the pipeline and implemented in the [Lane](Lane.py) class in method `draw_lines` in line #226 and `draw_test` in line #260. The method basically uses open-cv methods `cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))` for a green filled area between both detected lines, and `cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), thickness=20)` and `cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), thickness=20)` for left (red) and right (blue) line.

As previously mentioned, the perpective transform (birdeye) needs to be back-warped to the original image pane. We take use of the inverse matrix computed beforehand and call again the `cv2.warpPerspective` method from open-cv.

The overly effect is done using the `cv.addWeighted` method wich applies a slighlty transparent version of the lane detection to the original (though distorted) image. Here is an example of my result on a test image:

|Combined Birdeye|Combined Resulting Image| 
|---|---|
|![alt text][combbird]|![alt text][comb]|

---

###Pipeline (Video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1] for the project video. In the folder [output_videos](output_videos/) three more video can be found, however, the pipeline was designed to work with the project video and is not robust enough for all kinds of road conditions. As an additional gimmick I recorded an own image from my car during night time and applied a slighlty adjusted pipeline, in particular with changed source and destination points as the image size differs from the udacity videos.

To enhance quality of the lane finding for the video, a number of techniques and concepts were additional developed which can only be meaningfully aplied on a video stream. It essentailly makes use of previous states of the lines. For this the [Line](Line.py) class was enhanced with following fields:

```
self.detected               # Was the line detected in the last iteration? 
self.fit_ok                 # Was the fit evaluated as success?
self.all_coefficients       # Coefficients for the last n iterations
self.best_coefficients      # Polynomial coefficients averaged over the last n iterations  
self.current_coefficients   # Polynomial coefficients for the most recent fit
self.diffs                  # Difference in fit coefficients between last and new fits
self.radius_of_curvature    # Radius of curvature of the best fitted line in meter
self.best_x_offset          # Best x-offset
self.best_computed_x        # Best computed x values based on best coefficients     
self.line_base_pos          # Distance in meters of vehicle center from the line 
self.all_x_pixels           # X values for detected line pixels  
self.all_y_pixels           # Y values for detected line pixels
self.frame_number           # Current frame number
self.last_frame_update      # Frame number if last successful update
```

It would not fit into the context of all details of additonal approaches, but a summarized idea of the concept is given as follows:
* Smoothing coefficients: Each successfull fit is stored for the last n (which happens to be set to 5) frames in the `all_coefficients` variable. It is then smoothed (simple mean) over those iterations and finally stored inside the `best_coefficients` variable.
* Assessing line fit: This turned out to be not easily done and is very sensitive to configuration parameters. I basically compare the highest order coefficient (the one effecting the x^2 part of the polynomial) of the best fit with the current fit. If those deviate for more than a certain value (empirically this was set to `kThresholdX2 = 1E-5`), the fit was marked as unsucessful.
* Dismissing incorrect fits: Fits that have been marked as unsuccessful are skipped for this frame and the lane is not updated, but essentially the smoothed best fit is used instead. This however needs tracking of how many time we already dismissed a frame, since the assess part is not that robust and might mark an valid fit as unsuccessful.
* Using other line's coefficients: When a bad fit has been observerd for one line, the other uses for a number of frames the higher order coefficients of the other line.
* Retriggering histogram search: If bad fits have been reported over a period of time, we retrigger the full histogram search, just as we have done during the initial start of the pipeline.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Udacity already provided a good architectural idea of the pipeline which I used for my basic algorithm structure. As already described in the writeup, the pipeline consisted of the steps "Undistort the image", "Apply perpective transform", "Detect edges using thresholds (binary)", "Locate lane lines on the binary", "Fit both lines", "Check fit validity" and finally "Draw lane onto road". The techniques are already described in detail in the regarding section. 

For the image pipeline the extraction of line-points was most challenging and crucial, since we do not have any tracking possiblities for outlier detection. We must take the image for granted and tune the threshholds cor color and gradient as such a good results is achieved. The parameters hence work well with the test images (and the project video), but are not robust enough for the challenging video or even the harder-challenging video. The diversity of lightning, line colors, or even weather conditions expects a flexible setup of treshholds, there is not one soultion for all.

Also the pipeline is design for curvatures as appearing in the video and fails for the sharp curves in the hard-challenging video, as the area for perpective transform just is not wide enough. Next to these issues wich rather deal with tuning of parameters, there are systematical shortcoming in the pipeline, so that with simple pixel detection and fitting probably no good result for the challenging videos can be achieved.

For the project video the pipeline works quite alright with minor and slight jumps and somes unexact fits, but would not cause the car to drive completely off the lane. With help of the advanced techniques described earlier in the video pipeline section, a quite robust lane detection could be achieved. It has still potential for improvements, especially in the areas of assessing a fit of a line and the strategy behing re-triggering a full histogram search. Also the independence of both lines can be enhanced by i.e. only trigger a histo-search for one of the lines, sticking to the frame based approach for the other.
