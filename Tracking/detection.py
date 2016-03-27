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
        if position.is_correct():
            yield position
        is_end = False
        step_width = int(slip_step * position.width)
        step_height = int(slip_step * position.height)
        layer = 1
        xx = position.x
        yy = position.y
        while not is_end:
            is_end = True
            for start_point, vector in (([-1,-1],[1,0]),([1,-1],[0,1]),([1,1],[-1,0]),([-1,1],[0,-1])):
                position.update(x=xx + (start_point[0]*layer + vector[0])*step_width, y=yy+(start_point[1]*layer + vector[1])*step_height)
                while position.is_correct() and xx - layer*step_width <= position.x <= xx + layer*step_width and yy - layer*step_height <= position.y <= yy + layer*step_height:
                    is_end = False
                    yield position
                    position.update(x=position.x+vector[0]*step_width, y=position.y+vector[1]*step_height)
            layer =+ 1
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

    def detect(self, position, is_tracked):
        position = copy(position)
        detected_windows = []
        for current_position in scanning_window(position, scales_step = 1000, slip_step = 0.3, minimal_bounding_box_size = 50):
            patch = current_position.calculate_patch()
            result = self.cascaded_classifier(patch)
            if result == 1:
                detected_windows.append((current_position.get_window(), current_position.calculate_patch()))
                self.learning_component.add_new_positive(patch)
                if is_tracked:
                    return detected_windows
            else:
                self.learning_component.add_new_negative(patch)
        return detected_windows