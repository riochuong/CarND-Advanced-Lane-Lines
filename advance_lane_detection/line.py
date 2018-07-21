import numpy as np
class Line():
    def __init__(self, n=20):
        self.num_iteration_stored = n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self._ym_per_pix = 30/720

    def update(self, detected, allx=None, ally=None, fit_coeff=None):
        self.detected = detected
        # onl update data if line is detected
        if self.detected:
            assert allx is not None
            assert ally is not None
            assert fit_coeff is not None
            self.allx = allx
            self.ally = ally
            self.radius_of_curvature = self.calc_curvature()
            self.diffs = np.absolute(np.array(fit_coeff) - np.array(self.current_fit))
            self.current_fit = fit_coeff
            # revmove oldest element from the list
            if len(self.recent_xfitted) == self.num_iteration_stored:
                del self.recent_xfitted[0]
                self.recent_xfitted.append(allx)
                self.bestx = np.average(np.array(self.recent_xfitted), axis=0)

    def calc_curvature(self):
        y_eval = np.max(self.ally)
        return ((1 + (2*best_fit[0]*y_eval*ym_per_pix + best_fit[1])**2)**1.5) / np.absolute(2*best_fit[0])
