import numpy as np
from collections import deque

class Line:
    def __init__(self,n=5):
        self.n = n # length of queue to store data
        self.n_buffered = 0 # number of fits in buffer
        self.detected = False # was the line detected in the last iteration?
        self.recent_xfitted = deque([],maxlen=n) # x values of the last n fits of the line
        self.avgx = None #average x values of the fitted line over the last n iterations
        self.recent_fit_coeffs = deque([],maxlen=n) # fit coeffs of the last n fits
        self.avg_fit_coeffs = None # polynomial coefficients averaged over the last n iterations
        self.line_fitx = [np.array([False])] # xvals of the most recent fit
        self.line_fit = [np.array([False])] # polynomial coefficients for the most recent fit
        self.allx = None # x values for detected line pixels
        self.ally = None # y values for detected line pixels
        self.ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        self.line_curverad = None # radius of curvature of the line in some units
        self.line_pos = None # origin (pixels) of fitted line at the bottom of the image
        self.line_base_pos = None # distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float') # difference in fit coefficients between last and new fits

    def set_line_base_pos(self):
        y_eval = max(self.ploty)
        self.line_pos = self.line_fit[0]*y_eval**2 + self.line_fit[1]*y_eval + self.line_fit[2]
        basepos = 640
        self.line_base_pos = (self.line_pos - basepos)*3.7/650.0

    def curvature(self):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        self.line_fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        self.line_curverad = ((1 + (2*self.line_fit_cr[0]*y_eval*ym_per_pix + self.line_fit_cr[1])**2)**1.5) / np.absolute(2*self.line_fit_cr[0])

    def get_diffs(self):
        if self.n_buffered>0:
            self.diffs = self.line_fit - self.avg_fit_coeffs
        else:
            self.diffs = np.array([0,0,0], dtype='float')

    def accept_lane(self):
        flag = True
        maxdist = 2.8  # distance in meters from the lane
        if(abs(self.line_base_pos) > maxdist ):
            print(self.line_base_pos)
            print('lane too far away')
            flag  = False
        if(self.n_buffered > 0):
            relative_delta = self.diffs / self.avg_fit_coeffs
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta)<np.array([0.7,0.5,0.15])).all():
                # print('fit coeffs too far off [%]',relative_delta)
                flag=False
        return flag

    def add_data(self):
        self.recent_xfitted.appendleft(self.line_pos)
        self.recent_fit_coeffs.appendleft(self.line_fit)
        assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
        self.n_buffered = len(self.recent_xfitted)

    def pop_data(self):
        if self.n_buffered>0:
            self.recent_xfitted.pop()
            self.recent_fit_coeffs.pop()
            assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
            self.n_buffered = len(self.recent_xfitted)

        return self.n_buffered

    def set_avgx(self):
        fits = self.recent_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.avgx = avg

    def set_avgcoeffs(self):
        coeffs = self.recent_fit_coeffs
        if len(coeffs)>0:
            avg=0
            for coeff in coeffs:
                avg +=np.array(coeff)
            avg = avg / len(coeffs)
            self.avg_fit_coeffs = avg

    def update(self,lane):
        self.ally, self.allx = (lane[:,:,0]>254).nonzero()
        self.line_fit = np.polyfit(self.ally, self.allx, 2)
        self.line_fitx = self.line_fit[0]*self.ploty**2 + self.line_fit[1]*self.ploty + self.line_fit[2]
        self.set_line_base_pos()
        self.curvature()
        self.get_diffs()
        if self.accept_lane():
            self.detected=True
            self.add_data()
            self.set_avgx()
            self.set_avgcoeffs()
        else:
            self.detected=False
            self.pop_data()
            if self.n_buffered>0:
                self.set_avgx()
                self.set_avgcoeffs()

        return self.detected
