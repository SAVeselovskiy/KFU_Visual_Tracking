__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np
from learning import get_bounding_box

class Tracker:
    def __init__(self, learning_component):
        self.learning_component = learning_component
        # init_patch = learning_component.get_init_patch()

        # bounding_box = get_bounding_box(initial_frame, x, y, width, height)
        #
        # hsv_roi =  cv2.cvtColor(bounding_box, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        # self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        # cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        # self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def track(self, frame, track_window):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

        ret, track_window = cv2.meanShift(dst, track_window, self.term_crit)
        return track_window