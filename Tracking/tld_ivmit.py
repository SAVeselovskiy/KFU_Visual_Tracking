# -*- coding: utf-8 -*-
__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

from detection import Detector
from tracking import Tracker
from integration import Integrator, windows_intersection
from learning import LearningComponent
from structure import Position
import cv2
import numpy as np

from time import time

class TLD_IVMIT:
    def __init__(self, frame, window, init_frames_count = 100):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.position = Position(frame, *window)
        self.learning_component = LearningComponent(self.position.calculate_patch())
        self.detector = Detector(self.learning_component)
        self.tracker = Tracker(self.position)
        self.is_visible = True
        self.integrator = Integrator(self.learning_component)
        self.init_frames_count = init_frames_count
        self.detected_windows = None
        self.tracked_window = None

    def start(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.init_frames_count == 0:
            start = time()
            self.tracked_window = self.tracker.track(frame, self.position)
            if self.tracked_window is not None:
                self.position.update(frame, *self.tracked_window)
            print "Tracking:", time()- start

            start = time()
            self.detected_windows = self.detector.detect(self.position, self.tracked_window is not None)
            print "Detected windows count:", len(self.detected_windows)
            print "Detection:", time()- start

            start = time()
            filtered_detected_windows = [(window, patch, proba) for window, patch, proba in self.detected_windows if proba > 0.5]
            single_window, self.is_visible = self.integrator.get_single_window(self.position, filtered_detected_windows, self.tracked_window)
            print "Integration:", time()- start

            # start = time()
            # self.learning_component.n_expert()
            # self.learning_component.p_expert()
            # print "Update training set:", time()- start
            print
        else:
            self.tracked_window = self.tracker.track(frame, self.position)
            if self.tracked_window is not None:
                i = 0
                while i < 5:
                    self.position.update(x=np.random.randint(0,frame.shape[1]-self.position.width))
                    if self.position.is_correct() and windows_intersection(self.position.get_window(), self.tracked_window) == 0:
                        self.learning_component.update_negatives(self.position.calculate_patch())
                        i += 1

                self.position.update(frame, *self.tracked_window)
                self.learning_component.update_positives(self.position.calculate_patch())

                self.init_frames_count -= 1
            else:
                self.init_frames_count = 0
                self.is_visible = False

        return self.position
