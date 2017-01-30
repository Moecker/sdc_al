import numpy as np
import cv2

from Line import Line
import plotting as plot
import udacity_code as udacity


class Lane:
    def __init__(self, log, name="", is_debug=False, use_history=False):
        self.name = name
        # Constant to define extraction behavior
        self.kNumberOfSlidingWindows = 8
        self.kExtractWindowSize = int(320 / self.kNumberOfSlidingWindows)
        self.kSearchWindowSize = int(2 * self.kExtractWindowSize)
        # Logging interface
        self.log = log
        # Debug mode flag
        self.is_debug = is_debug
        # The left line
        self.left_line = Line(name=self.name + "_left", log=log, is_debug=is_debug)
        # The right line
        self.right_line = Line(name=self.name + "_right", log=log, is_debug=is_debug)
        # Keep track of history
        self.use_history=use_history
        # Has a lane been detected previousely?
        self.was_detected=False
        

    def locate_lines_udacity(self, binary_warped):
        udacity.locate_lines(binary_warped)
       

    def locate_lines_frame_based_udacity(self, binary_warped):
        udacity.find_lines_from_previous(binary_warped, 
                                         self.left_line.best_coefficients,
                                         self.right_line.best_coefficients)


    def locate_lines(self, img):
        self.log.debug("Locating lines ...")

        # The left and right binary which only contain line points
        left_extracted = np.zeros_like(img)
        right_extracted = np.zeros_like(img)

        if self.was_detected:
            self._locate_lines_frame_based_(img, left_extracted, right_extracted)
        else:
            self._locate_lines_histo_based_(img, left_extracted, right_extracted)
            self.was_detected = True
       

        if self.is_debug: 
            plotted = plot.plot_images(left_extracted, right_extracted, self.name + "_both_extracted", is_gray=True)

        # From the image only get the non zero element (which will be our to-be-fitted points)
        self._extract_fit_points_(left_extracted, self.left_line)
        self._extract_fit_points_(right_extracted, self.right_line)
      

    def _locate_lines_frame_based_(self, img, left_extracted, right_extracted):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = self.left_line.best_coefficients
        right_fit = self.right_line.best_coefficients

        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_extracted[lefty, leftx] = 1
        right_extracted[righty, rightx] = 1


    def _locate_lines_histo_based_(self, img, left_extracted, right_extracted):
        prev_left, prev_right, dist = None, None, lambda:0

        # For each sliding window use histogram technique to find the possible line segments
        for window_index in range(self.kNumberOfSlidingWindows, 0, -1):
            kInitSearchWindow = 2
            # For initial search use the lower kInitSearchWindow half
            if window_index == self.kNumberOfSlidingWindows:
                start_y = int(img.shape[0])
                end_y = int(img.shape[0] / kInitSearchWindow)   
            else:
                start_y = int(window_index * img.shape[0] / self.kNumberOfSlidingWindows)
                end_y = int((window_index - 1) * img.shape[0] / self.kNumberOfSlidingWindows)

            img_section = img[end_y:start_y,:]
            histogram = np.sum(img_section, axis=0)

            # Find X coords of two peaks of histogram, either full (for first window) ...
            if prev_left == None or prev_right == None:
                left_x, right_x = self.full_histogram_search(histogram)
                dist.left, dist.right = 0, 0
            # ... or based on the previous peaks
            else:
                left_x = self.windowed_histogram_search(histogram, prev_left)
                right_x = self.windowed_histogram_search(histogram, prev_right)

                dist.left, dist.right = np.abs(prev_left - left_x), np.abs(prev_right - right_x)

            # Add the target window from end to start per line
            if dist.left < self.kExtractWindowSize:
                left_extracted[end_y:start_y, :] = self._extract_line_image_(img_section, left_x)
                # Update previous left points
                prev_left = left_x

            if dist.right < self.kExtractWindowSize:
                right_extracted[end_y:start_y, :] = self._extract_line_image_(img_section, right_x)
                # Update previous right points
                prev_right = right_x


    def _extract_fit_points_(self, img, line):
        self.log.debug("Extracting fit points ...")
        all_non_zero = np.nonzero(img)
        line.all_x_pixels = all_non_zero[1]
        line.all_y_pixels = all_non_zero[0]
  

    def _extract_line_image_(self, img_section, x):
        mask = np.zeros_like(img_section)
        mask[:, (x - self.kExtractWindowSize):(x + self.kExtractWindowSize)] = 1
        mask = (mask==1)
        return img_section & mask


    def full_histogram_search(self, histogram):
        self.log.debug("Full histogram search ...")
        num_pixels = len(histogram)

        half_width = int(num_pixels/2)
        left_peak_x = np.argmax(histogram[0:half_width])
        right_peak_x = half_width + np.argmax(histogram[half_width:])

        return left_peak_x, right_peak_x


    def windowed_histogram_search(self, histogram, prev_x):
        self.log.debug("Windowed histogram search ...")

        start = max(0, int(prev_x - (self.kSearchWindowSize/2)))
        end = min(len(histogram), int(prev_x + (self.kSearchWindowSize/2)))
        peak_x = start + np.argmax(histogram[start:end])

        return peak_x


    def fit_lines(self):
        # Fit a second order polynomial to pixel positions in each fake lane line
        self.left_line.fit(self.use_history)
        self.right_line.fit(self.use_history)

    
    def draw_lines(self, image, binary_birdeye, birdeye_image, inverse_M):
        self.log.debug("Drawing lines ...")

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_birdeye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack(
            [self.left_line.best_computed_x, self.left_line.all_y_pixels]))])

        pts_right = np.array([np.flipud(np.transpose(np.vstack(
            [self.right_line.best_computed_x, self.right_line.all_y_pixels])))])

        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the transformed_image blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        cv2.polylines(color_warp, np.int_([pts_left]), False, (0,0,255), thickness=20)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (255,0,0), thickness=20)

        # Warp the blank back to original image space using inverse perspective matrix (inverse_M)
        newwarp = cv2.warpPerspective(color_warp, inverse_M, (image.shape[1], image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)
        result_bird = cv2.addWeighted(birdeye_image, 1, color_warp, 0.5, 0)

        self._draw_text_(result)
  
        return result, result_bird


    def _draw_text_(self, result):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Write the radius of curvature for each lane 
        left_roc = "Curv: {0:.2f}m".format(self.left_line.radius_of_curvature) 
        cv2.putText(result, left_roc, (10, 650), font, 1, (255, 255, 255), 2)

        right_roc = "Curv: {0:.2f}m".format(self.right_line.radius_of_curvature) 
        cv2.putText(result, right_roc, (1000, 650), font, 1, (255, 255, 255), 2)
    
        # Write the x coords for each lane 
        left_coord = "X   : {0:.2f}px".format(self.left_line.best_x_offset) 
        cv2.putText(result, left_coord, (10, 700), font, 1, (255, 255, 255), 2)

        right_coord = "X   : {0:.2f}px".format(self.right_line.best_x_offset) 
        cv2.putText(result, right_coord, (1000, 700), font, 1, (255, 255, 255), 2)

        # Write dist from center
        perfect_center = result.shape[1] / 2.0
        lane_x = self.right_line.best_x_offset - self.left_line.best_x_offset
        center_x = (lane_x / 2.0) + self.left_line.best_x_offset

        # According to US regulation the lane width is 3.70m
        kCmPerPixel = 3.70 / lane_x
        dist_from_center = (center_x - perfect_center) * kCmPerPixel

        dist_text = "Dist from Center: {0:.2f} m".format(dist_from_center)
        cv2.putText(result, dist_text, (450,50), font, 1, (255,255,255), 2)