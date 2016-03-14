__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent, get_bounding_box
from time import time

def get_frame(cap, fx_size = 0.15, fy_size = 0.15):
    if cap is not None:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = fx_size, fy = fy_size)
        return frame
    else:
        return None

class TLD_IVMIT:
    def __init__(self, cap, window_name, initial_frame, x, y, width, height):
        self.x = x
        self.y = y
        self.cap = cap
        self.window_name = window_name
        self.width_bounding_box = width
        self.height_bounding_box = height
        self.learning_component = LearningComponent(initial_frame, x, y, width, height)
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker(initial_frame, x, y, width, height)
        self.is_visible = True
        self.integrator = Integrator()

    def start(self):
        while True:
            frame = get_frame(self.cap)
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            last_time = time()
            detected_windows = self.detector.detect(gray_frame, self.width_bounding_box, self.height_bounding_box)
            tracked_window = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
            single_window = self.integrator.get_single_window(gray_frame, detected_windows, tracked_window, self.detector.nearest_neighbor_classifier)
            if single_window != None:
                self.x, self.y, self.width_bounding_box, self.height_bounding_box = single_window
            else:
                self.is_visible = False
            self.learning_component.n_expert()
            self.learning_component.p_expert()
            current_time = time()

            gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
            # for x, y, width, height in detected_windows:
            #     cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

            x, y, width, height = tracked_window
            cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))

            if self.is_visible:
                cv2.rectangle(gray_bgr_frame, pt1 = self.point_left_up(), pt2 = self.point_right_down(), color = (255, 0, 0))

            fps = int(100 / (current_time - last_time)) / 100.0
            if fps < 25:
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.putText(gray_bgr_frame, str(fps), (10, frame.shape[0]-10), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
            # print int(10/(current_time - last_time))/10.0, 'FPS ', current_time - last_time, 'Sec'

            cv2.imshow(self.window_name,gray_bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def point_left_up(self):
        return (self.x, self.y)

    def point_right_down(self):
        return (self.x + self.width_bounding_box, self.y + self.height_bounding_box)
