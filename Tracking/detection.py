__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class PatchVarianceClassifier:
    def __init__(self, init_patch):
        self.init_patch_variance = np.var(init_patch)

    def classify(self, patch):
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        if np.var(patch) > 0.5 * self.init_patch_variance:
            return 1
        else:
            return 0

class EnsembleClassifier:
    def __init__(self, learning_component):
        self.learning_component = learning_component
        self.classifier = RandomForestClassifier()

    def classify(self, patch):
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        feature = self.learning_component.get_feature(patch)
        return self.classifier.predict(feature)

    def relearn(self):
        x, y = self.learning_component.get_training_set()
        self.classifier.fit(x, y)

class NearestNeighborClassifier:

    def __init__(self, learning_component, lmbd = 0.1, tetta = 0.6):
        self.learning_component = learning_component
        self.lmbd = lmbd
        self.tetta = tetta

    def classify(self, patch):
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        if self.learning_component.relative_similarity(patch) > self.tetta:
            return 1
        else:
            return 0

def scanning_window(image, bounding_box_size, scales_step = 1.2, horizontal_step = 0.1, vertical_step = 0.1, minimal_bounding_box_size = 20):
    height, width = bounding_box_size
    flag_inc = True
    flag_dec = False
    while min(width, height) >= minimal_bounding_box_size:
        for y in xrange(0, image.shape[0]-height, int(vertical_step * height)):
            for x in xrange(0, image.shape[1]-width, int(horizontal_step * width)):
                yield (x, y, get_bounding_box(image, x, y, width, height))
                # clone = image.copy()
                # cv2.rectangle(clone, (x, y), (x + window_size[1], y + window_size[0]), (0, 255, 0), 2)
                # cv2.imshow("TLP_IVMIT", clone)
                # cv2.waitKey(1)
        if flag_inc:
            height = int(height * scales_step)
            width = int(width * scales_step)
        if flag_dec:
            height = int(height / scales_step)
            width = int(width / scales_step)
        if height > image.shape[0] or width > image.shape[1]:
            flag_inc = False
            flag_dec = True

class Detector:
    def __init__(self, learning_component, position):
        self.learning_component = learning_component
        self.patch_variance_classifier = PatchVarianceClassifier(learning_component.init_patch)
        self.ensemble_classifier = EnsembleClassifier(learning_component)
        # self.nearest_neighbor_classifier = NearestNeighborClassifier(learning_component)

    def cascaded_classifier(self, patch):
        # 3 stages of classify
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        if self.patch_variance_classifier.classify(patch) == 0:
            return 0
        else:
            return self.ensemble_classifier.classify(patch)

        # else:
        #     return self.nearest_neighbor_classifier.classify(patch)

    def detect(self, frame, position):
        bounding_box_size = np.array([position.height, position.width])
        detected_windows = []
        for x, y, bounding_box in scanning_window(frame, bounding_box_size, scales_step = 1.5, horizontal_step = 0.3, vertical_step = 0.3, minimal_bounding_box_size = 50):
            patch = cv2.resize(bounding_box, (15,15))
            result = self.cascaded_classifier(patch)
            if result == 1:
                detected_windows.append(((x, y, bounding_box.shape[1], bounding_box.shape[0]), patch))
                self.learning_component.add_new_positive(patch)
            else:
                self.learning_component.add_new_negative(patch)
        return detected_windows