import numpy as np
import cv2

from Line import Line
import plotting as plot


class Lane:
    def __init__(self, log, is_debug=False):
        # Logging interface
        self.log = log
        # Debug mode flag
        self.is_debug = is_debug
        # The left line
        self.left_line = Line(name="left", log=log, is_debug=is_debug)
        # The right line
        self.right_line = Line(name="right", log=log, is_debug=is_debug)
        
        
    def locate_lines(self, img):
        self.log.debug("Locating lines ...")
        number_of_sliding_windows = 4

        left_extracted = np.zeros_like(img)
        right_extracted = np.zeros_like(img)

        prev_left, prev_right = None, None

        for window_index in range(number_of_sliding_windows, 0, -1):
            start_y = int(window_index * img.shape[0] / number_of_sliding_windows)
            end_y = int((window_index - 1) * img.shape[0] / number_of_sliding_windows)

            img_section = img[end_y:start_y,:]
            histogram = np.sum(img_section, axis=0)

            # Find X coords of two peaks of histogram
            if prev_left == None or prev_right == None:
                left_x, right_x = self.full_histogram_search(histogram)
            else:
                left_x = self.windowed_histogram_search(histogram, prev_left)
                right_x = self.windowed_histogram_search(histogram, prev_right)

            left_extracted[end_y:start_y, :] = self._extract_line_image_(img_section, left_x)
            right_extracted[end_y:start_y, :] = self._extract_line_image_(img_section, right_x)

            prev_left, prev_right = left_x, right_x

        if self.is_debug: plotted = plot.plot_images(left_extracted, right_extracted, "left_right", is_gray=True)

        self._extract_fit_points_(left_extracted, self.left_line)
        self._extract_fit_points_(right_extracted, self.right_line)
      

    def _extract_fit_points_(self, img, line):
        self.log.debug("Extracting fit points ...")
        all_non_zero = np.nonzero(img)
        line.all_x_pixels = all_non_zero[1]
        line.all_y_pixels = all_non_zero[0]
  

    def _extract_line_image_(self, img_section, x):
        windows_size = 40

        mask = np.zeros_like(img_section)
        mask[:, (x - windows_size):(x + windows_size)] = 1
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
        kHistWindow = 160

        start = max(0, int(prev_x - (kHistWindow/2)))
        end = min(len(histogram), int(prev_x + (kHistWindow/2)))
        peak_x = start + np.argmax(histogram[start:end])

        return peak_x


    def fit_lines(self):
        # Fit a second order polynomial to pixel positions in each fake lane line
        self.left_line.fit()
        self.right_line.fit()

    
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
  
        return result, result_bird
