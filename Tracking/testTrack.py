__author__ = 'sergejveselovskij'

import cv2
import numpy as np
from tld_ivmit import TLD_IVMIT, get_frame

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
point1 = None
point2 = None
cap = cv2.VideoCapture(0)
flag = False
color = np.random.randint(0,255,(100,3))
def reset_tracking(event,x,y,flags,param):
    global point1, point2, tld

    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        point2 = None

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        point2 = (x, y)

    if event == cv2.EVENT_LBUTTONUP:
        frame = get_frame(cap)
        left_x = min(point1[0],point2[0])
        right_x = max(point1[0],point2[0])
        up_y = min(point1[1],point2[1])
        down_y = max(point1[1],point2[1])
        point1 = None
        point2 = None


def start():
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)
    while 1:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = None #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # img = cv2.add(frame,mask)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.setMouseCallback('frame', reset_tracking)
while flag:
    frame = get_frame(cap)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
    if point1 != None and point2 != None:
        cv2.rectangle(gray_bgr_frame, pt1 = point1, pt2 = point2, color = (30, 200, 120))
    cv2.imshow('frame',gray_bgr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
start()

cap.release()
cv2.destroyAllWindows()
