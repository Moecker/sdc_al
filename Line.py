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


    def fit(self, use_history=False):
        self.log.debug("Fitting line ...")

        # Compute current coefficients
        self.current_coefficients = np.polyfit(self.all_y_pixels, self.all_x_pixels, 2)
        self.all_coefficients.append(self.current_coefficients)
        
        # Compute mean if we have more than one fit and history is active
        if len(self.all_coefficients) > 1 and use_history:
            self.best_coefficients = np.mean(self.all_coefficients, axis=0)
            self.diffs = self.best_coefficients - self.current_coefficients
        else:        
            self.best_coefficients = self.all_coefficients[0]

        # Compute the x values based on the bes fit
        coef = self.best_coefficients
        self.best_computed_x = coef[0] * self.all_y_pixels**2 + coef[1] * self.all_y_pixels + coef[2]

        # Extract the curvature
        self._compute_curvature_()

        # Add top-most and bottom-most point to our y-array
        self.fill_gap()

        # Get the x-offset
        self.best_x_offset = self.best_computed_x[-1]

        self._plot_fit_()


    def _compute_curvature_(self):
        coef = self.best_coefficients
        y_eval = 300

        self.radius_of_curvature = ((1 + (2*coef[0] * y_eval + coef[1])**2)**1.5) / np.absolute(2*coef[0])


    def _plot_fit_(self):
        if self.is_debug:
            plt.clf()
            plt.plot(self.all_x_pixels, self.all_y_pixels[1:-1], 'o', color='red', markersize=2)

            plt.xlim(0, 1280)
            plt.plot(self.best_computed_x, self.all_y_pixels, color='green', linewidth=4)

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


