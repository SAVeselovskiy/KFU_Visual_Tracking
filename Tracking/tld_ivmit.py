__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

from detection import Detector
from tracking import Tracker
from integration import Integrator
from learning import LearningComponent

class TLD_IVMIT:
    def __init__(self, init_position, init_frames_count = 150):
        self.position = init_position
        self.learning_component = LearningComponent(self.position)
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker()
        self.is_visible = True
        self.integrator = Integrator()
        self.init_frames_count = init_frames_count

    def start(self, frame):
        if self.init_frames_count == 0:
            pass
            # detected_windows = self.detector.detect(frame, self.width_bounding_box, self.height_bounding_box)
            # tracked_window = self.tracker.track(frame, (self.x, self.y, self.width_bounding_box, self.height_bounding_box))
            # single_window = self.integrator.get_single_window(frame, detected_windows, tracked_window, self.detector.nearest_neighbor_classifier)
            # if single_window != None:
            #     self.x, self.y, self.width_bounding_box, self.height_bounding_box = single_window
            # else:
            #     self.is_visible = False
            # self.learning_component.n_expert()
            # self.learning_component.p_expert()

        else:
            (x, y, w, h) = self.tracker.track(frame, self.position)
            self.position.update(frame, x, y, w, h)
            self.learning_component.update_positives(self.position)
            self.init_frames_count -= 1

        return self.position
