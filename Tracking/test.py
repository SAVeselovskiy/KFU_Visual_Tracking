__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkyi Sergei'

import cv2
from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent, get_bounding_box

class TLD_IVMIT:
    def __init__(self, initial_frame, x, y, width, height):
        initial_bounding_box = get_bounding_box(initial_frame, x, y, width, height)
        self.x = x
        self.y = y
        self.width_bounding_box = width
        self.height_bounding_box = height
        self.learning_component = LearningComponent(initial_frame, x, y, width, height)
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker(initial_bounding_box)
        self.is_visible = True
        self.integrator = Integrator()

    def compute(self, frame):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detected_windows = self.detector.detect(gray_frame, self.width_bounding_box, self.height_bounding_box)
        tracked_window = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
        single_window = self.integrator.get_single_window(gray_frame, detected_windows, tracked_window, self.detector.nearest_neighbor_classifier)
        if single_window is not None:
            self.x, self.y, self.width_bounding_box, self.height_bounding_box = single_window
        else:
            self.is_visible = False
        self.learning_component.n_expert()
        self.learning_component.p_expert()

    def point_left_up(self):
        return (self.x, self.y)

    def point_right_down(self):
        return (self.x + self.width_bounding_box, self.y + self.height_bounding_box)

__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkyi Sergei'

import cv2
from tld_ivmit import TLD_IVMIT
from time import time

point1 = None
point2 = None
cap = cv2.VideoCapture(0)
window_name = 'TLD_IVMIT'

def get_frame(cap, fx_size = 0.2, fy_size = 0.2):
    if cap is not None:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = fx_size, fy = fy_size)
        return frame
    else:
        return None


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
        tld = TLD_IVMIT(frame, left_x, up_y, right_x-left_x, down_y-up_y)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, reset_tracking)

tld = None
while tld is None:
    frame = get_frame(cap)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
    if point1 != None and point2 != None:
        cv2.rectangle(gray_bgr_frame, pt1 = point1, pt2 = point2, color = (30, 200, 120))
    cv2.imshow(window_name,gray_bgr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while tld is not None:
    frame = get_frame(cap)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    last_time = time()
    tld.compute(frame)
    current_time = time()

    gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
    # for x, y, width, height in detected_windows:
    #     cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

    # x, y, width, height = tracked_window
    # cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

    cv2.rectangle(gray_bgr_frame, pt1 = tld.point_left_up(), pt2 = tld.point_right_down(), color = (255, 0, 0))

    fps = int(100 / (current_time - last_time)) / 100.0
    if fps < 25:
        color = (0,0,255)
    else:
        color = (0,255,0)
    cv2.putText(gray_bgr_frame, str(fps), (10, frame.shape[0]-10), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
    # print int(10/(current_time - last_time))/10.0, 'FPS ', current_time - last_time, 'Sec'

    cv2.imshow(window_name,gray_bgr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
