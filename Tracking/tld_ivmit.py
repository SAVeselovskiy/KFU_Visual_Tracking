__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent

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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # (x, y, w, h) = self.tracker.track(self.current_position.frame)
        # self.current_position.update(frame, x, y, w, h)
        #
        # count -= 1
        # if count > 0:
        #     detected_windows = self.detector.detect(gray_frame, self.width_bounding_box, self.height_bounding_box)
        #     tracked_window = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
        #     single_window = self.integrator.get_single_window(gray_frame, detected_windows, tracked_window, self.detector.nearest_neighbor_classifier)
        #     if single_window != None:
        #         self.x, self.y, self.width_bounding_box, self.height_bounding_box = single_window
        #     else:
        #         self.is_visible = False
        #     self.learning_component.n_expert()
        #     self.learning_component.p_expert()
        #
        # else:
        #     x, y, width, height = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
        #     self.learning_component.update_positives(get_bounding_box(frame, x, y, width, height))

        return self.current_position
