__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent
from structure import Position
import cv2

class TLD_IVMIT:
    def __init__(self, frame, window, init_frames_count = 150):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.position = Position(frame, *window)
        self.learning_component = LearningComponent(self.position)
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker()
        self.is_visible = True
        self.integrator = Integrator()
        self.init_frames_count = init_frames_count

    def start(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.init_frames_count == 0:
            detected_windows = self.detector.detect(frame, self.position)
            tracked_window = self.tracker.track(frame, self.position)

            single_window = self.integrator.get_single_window(frame, detected_windows, tracked_window)
            if single_window != None:
                self.position.update(frame, *single_window)
            else:
                self.is_visible = False
            self.learning_component.n_expert()
            self.learning_component.p_expert()
            self.detector.ensemble_classifier.relearn()

        else:
            tracked_window = self.tracker.track(frame, self.position)
            if tracked_window != None:
                self.position.update(frame, *tracked_window)
                self.learning_component.update_positives(self.position)
                self.init_frames_count -= 1
            else:
                self.init_frames_count = 0
                self.is_visible = False

        return self.position
