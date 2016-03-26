__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def track(self, frame, last_position):
        bounding_box = last_position.get_bounding_box()
        bounding_box =  cv2.cvtColor(bounding_box, cv2.COLOR_GRAY2BGR)
        hsv_roi =  cv2.cvtColor(bounding_box, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.meanShift(dst, (last_position.x, last_position.y, last_position.width, last_position.height), self.term_crit)
        return track_window