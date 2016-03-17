__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent
from time import time

class TLD_IVMIT:
    def __init__(self, current_position, init_frames_count = 150):
        self.current_position = current_position
        self.learning_component = LearningComponent(self.current_position)
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker(self.learning_component)
        self.is_visible = True
        self.integrator = Integrator()
        self.init_frames_count = init_frames_count

    def start(self, frame):
        pass
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # last_time = time()
        # (x, y, w, h) = self.tracker.track(self.current_position.frame)
        # self.current_position.update(frame, x, y, w, h)
        #
        # # count -= 1
        # # if count > 0:
        # #     detected_windows = self.detector.detect(gray_frame, self.width_bounding_box, self.height_bounding_box)
        # #     tracked_window = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
        # #     single_window = self.integrator.get_single_window(gray_frame, detected_windows, tracked_window, self.detector.nearest_neighbor_classifier)
        # #     if single_window != None:
        # #         self.x, self.y, self.width_bounding_box, self.height_bounding_box = single_window
        # #     else:
        # #         self.is_visible = False
        # #     self.learning_component.n_expert()
        # #     self.learning_component.p_expert()
        # #
        # # else:
        # #     x, y, width, height = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
        # #     self.learning_component.update_positives(get_bounding_box(frame, x, y, width, height))
        # current_time = time()
        #
        # gray_bgr_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
        # # for x, y, width, height in detected_windows:
        # #     cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))
        #
        # # x, y, width, height = tracked_window
        # # cv2.rectangle(gray_bgr_frame, pt1 = (x, y), pt2 = (x+width, y+height), color = (0, 0, 255))
        #
        # if self.is_visible:
        #     cv2.rectangle(gray_bgr_frame, pt1 = self.point_left_up(), pt2 = self.point_right_down(), color = (255, 0, 0))
        #
        # fps = int(100 / (current_time - last_time)) / 100.0
        # if fps < 25:
        #     color = (0,0,255)
        # else:
        #     color = (0,255,0)
        # cv2.putText(gray_bgr_frame, str(fps), (10, frame.shape[0]-10), cv2.FONT_ITALIC, 1, color, 2, cv2.LINE_AA)
        # # print int(10/(current_time - last_time))/10.0, 'FPS ', current_time - last_time, 'Sec'
        #
        # cv2.imshow(self.window_name,gray_bgr_frame)
