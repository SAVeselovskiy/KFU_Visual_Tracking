__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

from copy import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from  time import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import train_test_split
from structure import Position

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

    def predict_patch(self, patch):
        return np.var(patch.content) / self.init_patch_variance

    def predict_position(self, position):
        return np.var(position.calculate_patch().content) / self.init_patch_variance

class EnsembleClassifier:
    def __init__(self, learning_component):
        self.learning_component = learning_component
        self.classifier = RandomForestClassifier(max_depth=3)

    def classify(self, patch):
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        feature = patch.calculate_feature(self.learning_component.descriptor)
        if self.classifier.predict_proba(feature)[0][self.positive_class_index] > 0.5:
            return 1
        else:
            return 0

    def predict_patch(self, patch):
        feature = patch.calculate_feature(self.learning_component.descriptor)
        return self.classifier.predict_proba(feature)[0][self.positive_class_index]

    def predict_position(self, position):
        feature = position.calculate_patch().calculate_feature(self.learning_component.descriptor)
        return self.classifier.predict_proba(feature)[0][self.positive_class_index]

    def relearn(self, test_size=0):
        samples, weights, targets = self.learning_component.get_training_set(const_weight=True)
        train_samples, test_samples, train_targets, test_targets = train_test_split(samples, targets, test_size=test_size, random_state=np.random.RandomState(0))
        count_positives = 1.0*np.count_nonzero(train_targets)
        count_negatives = 1.0*(len(train_targets) - count_positives)
        positive_weight = count_negatives/len(train_targets)
        negative_weight = count_positives/len(train_targets)
        weights = np.array([positive_weight if target == 1 else negative_weight for target in train_targets])
        self.classifier.fit(train_samples, train_targets, sample_weight=weights)
        self.learning_component.new_samples_count = 0
        if len(test_samples) > 0:
            test_result = [self.classifier.predict(sample) for sample in test_samples]
            true_positives = 0.0
            count_test_positives = 1.0*np.count_nonzero(test_targets)
            count_result_positives = 1.0*np.count_nonzero(test_result)
            for i in xrange(len(test_targets)):
                if test_targets[i] == test_result[i] and test_result[i] == 1:
                    true_positives += 1
            precision = true_positives / count_test_positives
            recall = true_positives / count_result_positives
            print "Precision:", precision
            print "Recall", recall
            if precision + recall != 0:
                print "F-score:", 2 * precision * recall / (precision + recall)
            else:
                print "F-score:", 0
        self.positive_class_index = 0
        for elem in self.classifier.classes_:
            if elem != 1.0:
                self.positive_class_index += 1
            else:
                break

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

    def predict_patch(self, patch):
        return self.learning_component.relative_similarity(patch)

    def predict_position(self, position):
        return self.learning_component.relative_similarity(position.calculate_patch())

def scanning_window(position, scales_step = 1.2, slip_step = 0.1, minimal_bounding_box_size = 20, min_step=1, max_step=20):
    flag_inc = True
    flag_dec = False
    init_position = copy(position)
    while min(position.width, position.height) >= minimal_bounding_box_size:
        position.update(x=0,y=0)
        step_width = min(max(min_step,int(slip_step * position.width)),max_step)
        step_height = min(max(min_step,int(slip_step * position.height)),max_step)
        while position.is_correct():
            while position.is_correct():
                yield position
                position.update(x=position.x+step_width)
            position.update(x=0, y=position.y+step_height)
        # if position.is_correct():
        #     yield position
        # is_end = False
        # step_width = int(slip_step * position.width)
        # step_height = int(slip_step * position.height)
        # layer = 1
        # xx = position.x
        # yy = position.y
        # while not is_end:
        #     is_end = True
        #     for start_point, vector in (([-1,-1],[1,0]),([1,-1],[0,1]),([1,1],[-1,0]),([-1,1],[0,-1])):
        #         position.update(x=xx + (start_point[0]*layer + vector[0])*step_width, y=yy+(start_point[1]*layer + vector[1])*step_height)
        #         while position.is_correct() and xx - layer*step_width <= position.x <= xx + layer*step_width and yy - layer*step_height <= position.y <= yy + layer*step_height:
        #             is_end = False
        #             yield position
        #             position.update(x=position.x+vector[0]*step_width, y=position.y+vector[1]*step_height)
        #     layer += 1
        if flag_inc:
            position.update(height=int(position.height * scales_step), width = int(position.width * scales_step))
        if position.height > position.frame.shape[0] or position.width > position.frame.shape[0]:
            flag_inc = False
            flag_dec = True
            position = init_position
        if flag_dec:
            position.update(height=int(position.height / scales_step), width = int(position.width / scales_step))

def get_sliding_positions(init_position, buffer_frame=None, scales_step = 1.2, slip_step = 0.1, minimal_bounding_box_size = 20, min_step=2, max_step=2):
    sliding_positions = []
    flag_inc = True
    flag_dec = False
    position = copy(init_position)
    while min(position.width, position.height) >= minimal_bounding_box_size:
        position.update(x=0,y=0)
        step_width = min(max(min_step,int(slip_step * position.width)),max_step)
        step_height = min(max(min_step,int(slip_step * position.height)),max_step)
        while position.is_correct():
            while position.is_correct():
                insert_position = copy(position)
                insert_position.clean_frame()
                sliding_positions.append(insert_position)
                position.update(x=position.x+step_width)
            position.update(x=0, y=position.y+step_height)
        if flag_inc:
            position.update(height=int(position.height * scales_step), width = int(position.width * scales_step))
        if position.height > position.frame.shape[0] or position.width > position.frame.shape[0]:
            flag_inc = False
            flag_dec = True
            position = copy(init_position)
        if flag_dec:
            position.update(height=int(position.height / scales_step), width = int(position.width / scales_step))
    return sliding_positions

class Detector:
    def __init__(self, init_position, learning_component, threshold_patch_variance=0.5, threshold_ensemble=0.3, threshold_nearest_neighbor=0.6):
        self.learning_component = learning_component
        self.patch_variance_classifier = PatchVarianceClassifier(learning_component.init_patch)
        self.ensemble_classifier = EnsembleClassifier(learning_component)
        self.nearest_neighbor_classifier = NearestNeighborClassifier(learning_component)
        self.threshold_patch_variance = threshold_patch_variance
        self.threshold_ensemble = threshold_ensemble
        self.threshold_nearest_neighbor = threshold_nearest_neighbor
        self.sliding_positions = get_sliding_positions(init_position, self.buffer_frame, scales_step = 1.2, slip_step = 0.05, minimal_bounding_box_size = 50, min_step=2, max_step=10)

    def cascaded_classifier(self, patch):
        # 3 stages of classify
        # return 1 if object is positive detected
        # return 0 if object is negative detected
        if self.patch_variance_classifier.predict_patch(patch) < self.threshold_patch_variance:
            return 0
        elif self.ensemble_classifier.predict_patch(patch) < self.threshold_patch_variance:
            return 0
        elif self.nearest_neighbor_classifier.predict_patch(patch) < self.threshold_nearest_neighbor:
            return 0
        else:
            return 1

    def detect(self, position, is_tracked):
        position = copy(position)
        if self.learning_component.new_samples_count > 10:
            start = time()
            self.ensemble_classifier.relearn()
            print "Relearn:", time() - start
        detected_windows = []
        predict_times = []
        for current_position in self.sliding_positions:
            pass
            current_position.update(frame=position.frame)
            # start = time()
            # proba = self.predict_position(current_position)
            # predict_times.append(time() - start)
            # if proba > 0.5:
            #     detected_windows.append((current_position.get_window(), current_position.calculate_patch(), proba))
            #     self.learning_component.add_new_positive(current_position.calculate_patch())
            #     if is_tracked:
            #         return detected_windows
            # else:
            #     self.learning_component.add_new_negative(current_position.calculate_patch())
        # print "Analysed window count:", len(predict_times)
        # print "Max detection time:", np.max(predict_times)
        # print "Min detection time:", np.min(predict_times)
        # print "Mean detection time:", np.mean(predict_times)
        return detected_windows

    def predict_patch(self, patch):
        return self.cascaded_classifier(patch)

    def predict_position(self, position):
        return self.cascaded_classifier(position.calculate_patch())