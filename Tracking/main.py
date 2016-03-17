__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from tld_ivmit import TLD_IVMIT
from structure import CurrentPosition

point1 = None
point2 = None
tld = None
cap = cv2.VideoCapture(0)
window_name = 'TLD_IVMIT'

def get_frame(cap, fx_size = 0.3, fy_size = 0.3):
    if cap is not None:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = fx_size, fy = fy_size)
        frame = cv2.flip(frame, 1)
        return frame
    else:
        return None

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
        current_position = CurrentPosition(frame, left_x, up_y, right_x-left_x, down_y-up_y)
        tld = TLD_IVMIT(current_position)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, reset_tracking)

while True:
    frame = get_frame(cap)
    if tld is None:
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
        if point1 != None and point2 != None:
            cv2.rectangle(gray_bgr_frame, pt1 = point1, pt2 = point2, color = (30, 200, 120))
        cv2.imshow(window_name, gray_bgr_frame)
    else:
        tld.start(frame)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
