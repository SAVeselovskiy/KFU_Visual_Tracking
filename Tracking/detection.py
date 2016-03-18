__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np

class PatchVarianceClassifier:
    def __init__(self, initial_patch):
        self.initial_patch_variance = np.var(initial_patch)

    def classify(self, patch):
        # return 1 if object is positive detected
        # return -1 if object is negative detected
        if np.var(patch) > 0.5 * self.initial_patch_variance:
            return 1
        else:
            return -1

class EnsembleClassifier:
    base_classifiers = []
    comparison = 13

    def __init__(self, initial_patch):
        pass
        # self.generate_pixel_comparisons(initial_patch)

    def classify(self, patch):
        # return 1 if object is positive detected
        # return -1 if object is negative detected
        base_classifiers_results = [self.base_classify()]
        if np.mean(base_classifiers_results) > 0.5:
            return 1
        else:
            return -1

    def generate_pixel_comparisons(self, patch):
        blured = cv2.blur(patch, 3)
        self.base_classifiers = np.zeros(2 ** self.comparison)

class NearestNeighborClassifier:

    def __init__(self, learning_component, lmbd = 0.1, tetta = 0.6):
        self.learning_component = learning_component
        self.lmbd = lmbd
        self.tetta = tetta

    def classify(self, patch):
        # return 1 if object is positive detected
        # return -1 if object is negative detected
        if self.learning_component.relative_similarity(patch) > self.tetta:
            return 1
        else:
            return -1

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
    def __init__(self, learning_component):
        self.learning_component = learning_component
        # init_patch = learning_component.get_init_patch()
        # self.patch_variance_classifier = PatchVarianceClassifier(init_patch)
        # self.ensemble_classifier = EnsembleClassifier(init_patch)
        # self.nearest_neighbor_classifier = NearestNeighborClassifier(learning_component)

    def cascaded_classifier(self, patch):
        # 3 stages of classify
        # return 1 if object is positive detected
        # return -1 if object is negative detected
        if self.patch_variance_classifier.classify(patch) == -1:
            return -1
        elif self.ensemble_classifier.classify(patch) == 1:
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(patch, None)
            return 1
        else:
            return self.nearest_neighbor_classifier.classify(patch)

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