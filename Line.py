import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import numpy as np
import collections

import plotting as plot

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, log, name="", is_debug=False):
        # Logging interface
        self.log = log
        # Line name
        self.name = name
        # Debug flag
        self.is_debug = is_debug

        # Was the line detected in the last iteration?
        self.detected = False  

        self.fit_ok = False

        # Coefficients for the last n iterations
        self.all_coefficients = collections.deque(maxlen=5)
        # Polynomial coefficients averaged over the last n iterations
        self.best_coefficients = None  
        # Polynomial coefficients for the most recent fit
        self.current_coefficients = [np.array([False])]  
        # Difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        # Radius of curvature of the best fitted line in meter
        self.radius_of_curvature = None 

        # Best x-offset
        self.best_x_offset = None
        # Best computed x values based on best coefficients
        self.best_computed_x = None     

        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 

        # X values for detected line pixels
        self.all_x_pixels = None  
        # Y values for detected line pixels
        self.all_y_pixels = None
        self.y_vals_for_drawing = None

        self.frame_number = 0
        self.last_frame_update = 0


    def reset(self):
        self.all_coefficients.clear()   


    def fit(self, use_history=False):
        self.log.debug("Fitting line ...")

        kMaxMissingFrames = 5
        kMinNumberOfPixels = 10

        if len(self.all_y_pixels) < kMinNumberOfPixels or len(self.all_x_pixels) < kMinNumberOfPixels:
            self.log.warn("Too less pixels, skipping fit")
            return
            
        # Compute current coefficients
        self.current_coefficients = np.polyfit(self.all_y_pixels, self.all_x_pixels, 2)
        
        # Compute mean if we have more than one fit and history is active
        if len(self.all_coefficients) > 1 and use_history:
            self.best_coefficients = np.mean(self.all_coefficients, axis=0)
            self.diffs = np.abs(self.best_coefficients - self.current_coefficients)

            # Check whether the lane has a reasonably fit compared to last fit
            self.fit_ok = self.check_fit()
        else:        
            self.best_coefficients = self.current_coefficients
            # We must expect the first fit to be correct since there is no comparison possible
            self.fit_ok = True

        self.all_coefficients.append(self.current_coefficients)

        # Only update line if fit was ok
        if self.fit_ok:
            self.update_line()
        else:
            self.log.warn("Fit of " + self.name + " was not ok")

        if np.abs(self.frame_number - self.last_frame_update) >= kMaxMissingFrames:
            self.log.warn("Mark " + self.name + " as not detected")
            self.detected = False          


    def update_line(self):
        # Compute the x values based on the bes fit
        coef = self.best_coefficients
        self.best_computed_x = coef[0] * self.y_vals_for_drawing**2 + coef[1] * self.y_vals_for_drawing + coef[2]

        # Extract the curvature
        self._compute_curvature_()

        # Get the x-offset
        self.best_x_offset = self.best_computed_x[-1]

        self._plot_fit_()

        self.last_frame_update = self.frame_number


    def check_fit(self):
        kThresholdX2 = 1E-4

        if self.diffs[0] > kThresholdX2:
            return False
        else:
            return True


    def _compute_curvature_(self):
        y_eval = np.max(self.y_vals_for_drawing)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        curvature_coeffs = np.polyfit(self.y_vals_for_drawing * ym_per_pix, self.best_computed_x * xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.radius_of_curvature = \
            ((1 + (2 * curvature_coeffs[0]*y_eval * ym_per_pix + curvature_coeffs[1]) ** 2) ** 1.5) \
            / np.absolute(2 * curvature_coeffs[0])
        

    def _plot_fit_(self):
        if self.is_debug:
            plt.clf()
            plt.plot(self.all_x_pixels, self.all_y_pixels[1:-1], 'o', color='red', markersize=2)

            plt.xlim(0, 1280)
            plt.plot(self.best_computed_x, self.y_vals_for_drawing, color='green', linewidth=4)

            plt.gca().invert_yaxis()
            plot.save_plot(plt, self.name + "_fitted.png")


    def fill_gap(self):
        kBottom = 720
        kTop = 0

        coefficients = self.best_coefficients

        self.all_y_pixels = np.insert(self.all_y_pixels, 0, kTop)
        self.best_computed_x = np.insert(self.best_computed_x, 0, coefficients[2])

        self.all_y_pixels  = np.append(self.all_y_pixels, kBottom)
        self.best_computed_x = np.append(
            self.best_computed_x, coefficients[0] * kBottom**2 + coefficients[1] * kBottom + coefficients[2])


