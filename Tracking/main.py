__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from tld_ivmit import TLD_IVMIT
from structure import Position
from time import time

point1 = None
point2 = None
tld = None
cap = cv2.VideoCapture(0)
window_name = 'TLD_IVMIT'
color = (255, 0, 0)

def get_frame(cap, fx_size = 0.3, fy_size = 0.3):
    if cap is not None:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = fx_size, fy = fy_size)
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

        # for x, y, width, height in detected_windows:
        #     cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

        # x, y, width, height = tracked_window
        # cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

        if tld.is_visible:
            if tld.init_frames_count == 0:
                color = clr
            cv2.rectangle(frame, pt1 = current_position.point_left_up(), pt2 = current_position.point_right_down(), color = color)
        cv2.putText(frame, str(fps), (10, frame.shape[0]-10), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
