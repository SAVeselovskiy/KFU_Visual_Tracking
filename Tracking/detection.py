__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

from copy import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

class PatchVarianceClassifier:
    def __init__(self, init_patch):
        self.init_patch_variance = np.var(init_patch.content)

    def classify(self, patch):
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        if np.var(patch.content) > 0.5 * self.init_patch_variance:
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
        feature = patch.calculate_feature(self.learning_component.descriptor)
        return self.classifier.predict(feature)

    def relearn(self):
        samples, weights, targets = self.learning_component.get_training_set()
        self.classifier.fit(samples, targets, sample_weight=weights)

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

def scanning_window(position, scales_step = 1.2, slip_step = 0.1, minimal_bounding_box_size = 20):
    flag_inc = True
    flag_dec = False
    while min(position.width, position.height) >= minimal_bounding_box_size:
        while position.is_correct():
            position.update(y=position.y+int(slip_step * position.height))
            while position.is_correct():
                position.update(x=position.x+int(slip_step * position.width))
                if position.is_correct():
                    yield position
            position.update(x=0)
        position.update(y=0)
        if flag_inc:
            position.update(height=int(position.height * scales_step), width = int(position.width * scales_step))
        if flag_dec:
            position.update(height=int(position.height / scales_step), width = int(position.width / scales_step))
        if position.height > position.frame.shape[0] or position.width > position.frame.shape[0]:
            flag_inc = False
            flag_dec = True

class Detector:
    def __init__(self, learning_component):
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

    def detect(self, position):
        position = copy(position)
        detected_windows = []
        for current_position in scanning_window(position, scales_step = 1.5, slip_step = 0.3, minimal_bounding_box_size = 50):
            patch = current_position.calculate_patch()
            result = self.cascaded_classifier(patch)
            if result == 1:
                detected_windows.append((current_position.get_window(), current_position.calculate_patch()))
                self.learning_component.add_new_positive(patch)
            else:
                self.learning_component.add_new_negative(patch)
        return detected_windows