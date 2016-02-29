__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from tld_ivmit import TLD_IVMIT, get_frame

point1 = None
point2 = None
tld = None
cap = cv2.VideoCapture(0)
window_name = 'TLD_IVMIT'

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
        tld = TLD_IVMIT(cap, window_name, frame, left_x, up_y, right_x-left_x, down_y-up_y)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, reset_tracking)

while tld == None:
    frame = get_frame(cap)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
    if point1 != None and point2 != None:
        cv2.rectangle(gray_bgr_frame, pt1 = point1, pt2 = point2, color = (30, 200, 120))
    cv2.imshow(window_name,gray_bgr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tld.start()

cap.release()
cv2.destroyAllWindows()
