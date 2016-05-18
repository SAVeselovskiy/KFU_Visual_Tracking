# -*- coding: utf-8 -*-
__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np
from tld_ivmit import TLD_IVMIT
from time import time

point1 = None
point2 = None
tld = None
cap = cv2.VideoCapture(0)
window_name = 'TLD_IVMIT'
color = (255, 0, 0)

def get_frame(cap, area=320*240, dsize = None, fx_size = None, fy_size = None):
    if cap is not None:
        ret, frame = cap.read()
        if area is not None and area != 0:
            f_size = np.sqrt(1.0*area/(frame.shape[0]*frame.shape[1]))
            frame = cv2.resize(frame, None, fx=f_size, fy=f_size)
        elif dsize is not None:
            frame = cv2.resize(frame, dsize)
        elif fx_size is not None and fy_size is not None:
            frame = cv2.resize(frame, None, fx=fx_size, fy=fy_size)
        frame = cv2.flip(frame, 1)
        return frame
    else:
        return None

def get_fps(time1, time2):
    fps = int(100 / (time2 - time1)) / 100.0
    if fps < 25:
        color = (0,0,255)
    else:
        color = (0,255,0)
    return fps, color

def reset_tracking(event,x,y,flags,param):
    global point1, point2, tld

    if event == cv2.EVENT_LBUTTONDOWN:
        tld = None
        point1 = (x, y)
        point2 = None

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        point2 = (x, y)

    if event == cv2.EVENT_LBUTTONUP:
        frame = get_frame(cap)
        if point1 is not None and point2 is not None:
            left_x = min(point1[0],point2[0])
            right_x = max(point1[0],point2[0])
            up_y = min(point1[1],point2[1])
            down_y = max(point1[1],point2[1])
            point1 = None
            point2 = None
            tld = TLD_IVMIT(frame, (left_x, up_y, right_x-left_x, down_y-up_y))

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, reset_tracking)

while True:
    frame = get_frame(cap)
    if tld is None:
        if point1 != None and point2 != None:
            cv2.rectangle(frame, pt1 = point1, pt2 = point2, color = color)
    else:
        time1 = time()
        current_position = tld.start(frame)
        time2 = time()
        fps, clr = get_fps(time1, time2)

        if tld.detected_windows is not None:
            for (x, y, width, height), patch, proba in tld.detected_windows:
                if proba > 0.5:
                    cv2.rectangle(frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (255, 255, 0))

        if tld.tracked_window is not None:
            x, y, width, height = tld.tracked_window
            cv2.rectangle(frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (255, 0, 255))

        if tld.is_visible:
            if tld.init_frames_count == 0:
                color = clr
            cv2.rectangle(frame, pt1 = current_position.point_left_up(), pt2 = current_position.point_right_down(), color = color)

        if tld.init_frames_count > 0:
            cv2.putText(frame, str(tld.init_frames_count), (10, 30), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, str(fps), (10, frame.shape[0]-10), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    # if tld is not None and tld.detected_windows is not None:
    #     ima = np.zeros((frame.shape[0],frame.shape[1]))
    #     for (x, y, width, height), patch, proba in tld.detected_windows:
    #         if proba > 0:
    #             for i in xrange(x,x+width):
    #                 for j in xrange(y,y+height):
    #                     ima[j,i] = max(proba, ima[j,i])
    #     cv2.imshow('Colormap of detector', ima)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
