import numpy as np


def calc_curvature(ally, coeff_fit, ym_per_pix):
    """
        Calculate radius curvature of the lane
    """
    y_eval = np.max(ally)
    return ((1 + (2*coeff_fit[0]*y_eval*ym_per_pix + coeff_fit[1])**2)**1.5) / np.absolute(2*coeff_fit[0])

class Line():
    def __init__(self, img_shape, n=20, distance_from_center_threshold=0.15):
        self.img_shape = img_shape
        self.num_iteration_stored = n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.recent_fit = []
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
        self._ym_per_pix = 30/self.img_shape[0]
        self._xm_per_pix = 3.7/self.img_shape[1]
        self._count = 0
        self._distance_from_center_threshold = distance_from_center_threshold

    def _is_lane_detected(self, fit_coeff=None, xfit=None, yfit=None):
        # check if lane is within distance to center

        bottom_y = np.max(yfit)
        x_mid = self.img_shape[1] // 2
        print("Mid distance: %.2f" % (x_mid * self._xm_per_pix))
        idx = np.where(yfit == bottom_y)
        print("idx ", idx)
        assert len(idx) > 0
        distance_from_center = np.absolute(x_mid - xfit[idx[0]]) * self._xm_per_pix
        if self.line_base_pos is not None:
            error = np.absolute(distance_from_center - self.line_base_pos) / self.line_base_pos
            if (not distance_from_center > 0) or (error > self._distance_from_center_threshold):
                print ("Distance from center is too high %.2f error: %.2f " % (distance_from_center, error))
                return False
        else:
            self.line_base_pos = distance_from_center
        # check if curvature had been dramatical change
        new_curvature = calc_curvature(yfit, fit_coeff, self._ym_per_pix)
        # if self.radius_of_curvature is not None:
        #     diff_curv_err = np.absolute((new_curvature - self.radius_of_curvature) / np.float(self.radius_of_curvature))
        #     if diff_curv_err > 0.5:
        #         print ("Curvature changes sharply %.2f" % diff_curv_err)
        #         return False
        # check if new curvature is within 1km
        if new_curvature > 5000 or new_curvature < 100:
            print ("Curvature is too bad %.2f. " % (new_curvature))
            return False
        # everything is good
        return True

    def update(self, allx=None, ally=None, fit_coeff=None, xfit=None, yfit=None):

        self.detected = self._is_lane_detected(fit_coeff, xfit, yfit)

        # onl update data if line is detected
        if self.detected:
            self._count += 1
            assert allx is not None
            assert ally is not None
            assert fit_coeff is not None
            self.allx = allx
            self.ally = ally
            if (self.best_fit is None):
                self.best_fit = fit_coeff

            self.diffs = np.absolute(np.array(fit_coeff) - np.array(self.current_fit))
            self.current_fit = fit_coeff
            # revmove oldest element from the list
            if len(self.recent_xfitted) == self.num_iteration_stored:
                del self.recent_xfitted[0]
                del self.recent_fit[0]
            # keeps appending fitted data
            self.recent_xfitted.append(xfit)
            self.recent_fit.append(fit_coeff)
            self.bestx = np.average(np.array(self.recent_xfitted), axis=0)
            self.best_fit = np.average(np.array(self.recent_fit), axis=0)
            self.radius_of_curvature = calc_curvature(ally, self.best_fit, self._ym_per_pix)
            if self._count == self.num_iteration_stored:
                self._count = 0
