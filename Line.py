import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import numpy as np

import plotting as plot

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, log, name=None, is_debug=False):
        # Logging interface
        self.log = log
        # Line name
        self.name = name
        # Debug flag
        self.is_debug = is_debug
        # Was the line detected in the last iteration?
        self.detected = False  
        # X values of the last n fits of the line
        self.recent_x_fitted = [] 
        # Average x values of the fitted line over the last n iterations
        self.best_computed_x = None     
        # Polynomial coefficients averaged over the last n iterations
        self.best_coefficients = None  
        # Polynomial coefficients for the most recent fit
        self.current_coefficients = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # Difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # X values for detected line pixels
        self.all_x_pixels = None  
        # Y values for detected line pixels
        self.all_y_pixels = None


    def fit(self):
        self.log.debug("Fitting line ...")

        coefficients = np.polyfit(self.all_y_pixels, self.all_x_pixels, 2)
        computed_x = coefficients[0] * self.all_y_pixels**2 + coefficients[1] * self.all_y_pixels + coefficients[2]

        self.best_coefficients = coefficients
        self.best_computed_x = computed_x

        self.fill_gap()

        if self.is_debug:
            mark_size = 3
            plt.plot(self.all_x_pixels, self.all_y_pixels[1:-1], 'o', color='red', markersize=mark_size)

            plt.xlim(0, 1280)
            plt.plot(self.best_computed_x, self.all_y_pixels, color='green', linewidth=3)

            plt.gca().invert_yaxis()
            plot.save_plot(plt, self.name + "_fitted.png")
            plt.gca().invert_yaxis()


    def fill_gap(self):
        kBottom = 720
        kTop = 0

        coefficients = self.best_coefficients

        self.all_y_pixels = np.insert(self.all_y_pixels, 0, kTop)
        self.best_computed_x = np.insert(self.best_computed_x, 0, coefficients[2])

        self.all_y_pixels  = np.append(self.all_y_pixels, kBottom)
        self.best_computed_x = np.append(
            self.best_computed_x, coefficients[0] * kBottom**2 + coefficients[1] * kBottom + coefficients[2])


