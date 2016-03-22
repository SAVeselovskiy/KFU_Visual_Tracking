__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np

def add_gaussian_noise(bounding_box, mean, sigma):
    if bounding_box is not None:
        return bounding_box + np.random.normal(mean, sigma, bounding_box.shape)
    else:
        return None

class LearningComponent:

    def __init__(self, init_position):
        self.positives = []
        self.negatives = []
        self.new_positives = []
        self.new_negatives = []
        self.feature_positives = []
        self.feature_negatives = []
        self.update_positives(init_position)
        self.descriptor = cv2.HOGDescriptor()
        self.init_patch = self.get_patch(init_position)

    #     self.generate_training_examples(init_position)
    #
    # def generate_training_examples(self, gray_initial_frame, x, y, width, height, closest_count = 10, surround_count = 50, small_radius = None, big_radius = None, sigma = 5):
    #     if small_radius == None:
    #         small_radius = 0.1 * (width + height) / 2
    #     if big_radius == None:
    #         big_radius = (width + height) / 2
    #
    #     for i in xrange(closest_count):
    #         fi = i * np.pi / closest_count
    #         diff_x = x + int(np.cos(fi) * small_radius)
    #         diff_y = y + int(np.sin(fi) * small_radius)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y, width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)
    #
    #         if bounding_box is not None:
    #             bounding_box = cv2.resize(bounding_box, None, fx=1.01,fy=1.01)
    #             generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #             self.update_negatives(generalise)
    #
    #             bounding_box = cv2.resize(bounding_box, None, fx=0.99,fy=0.99)
    #             generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #             self.update_negatives(generalise)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x-max(int(0.01*width), 1), diff_y, width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x+max(int(0.01*width), 1), diff_y, width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y-max(int(0.01*height), 1), width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y+max(int(0.01*height), 1), width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)
    #
    #     for i in xrange(surround_count):
    #         fi = i * np.pi / closest_count
    #         diff_x = x + int(np.cos(fi) * big_radius)
    #         diff_y = y + int(np.sin(fi) * big_radius)
    #
    #         bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y, width, height)
    #         generalise = add_gaussian_noise(bounding_box, 0, sigma)
    #         self.update_negatives(generalise)

    def get_training_set(self):
        x = []
        y = []
        for feature in self.feature_positives:
            x.append(feature)
            y.append(1)
        for feature in self.feature_negatives:
            x.append(feature)
            y.append(0)
        return x, y

    def get_patch(self, position):
        patch = cv2.resize(position.get_bounding_box(), self.patch_size)
        patch = np.uint8(np.rint(patch))
        return patch

    def NCC(self, pi, pj):
        # pi, pj - patches
        CV_TM_CCOEFF_NORMED = 5
        try:
            x = cv2.matchTemplate(pi, pj, CV_TM_CCOEFF_NORMED)[0][0]
        except:
            x = 1
        return x

    def similarity(self, pi, pj):
        y = 0.5 * (self.NCC(pi, pj) + 1)
        return y

    def similarity_positive(self, p):
        if len(self.positives) > 0:
            similarity_list = [self.similarity(p,pi) for pi in self.positives]
            return max(similarity_list)
        else:
            return 1

    def similarity_negative(self, p):
        if len(self.negatives) > 0:
            similarity_list = [self.similarity(p,pi) for pi in self.negatives]
            return max(similarity_list)
        else:
            return 1

    def similarity_half_first_positive(self, p):
        if len(self.positives)/2+1 > 0:
            similarity_list = [self.similarity(p,pi) for pi in self.positives[:len(self.positives)/2+1]]
            return max(similarity_list)
        else:
            return 1

    def relative_similarity(self, p):
        divisor = self.similarity_positive(p) + self.similarity_negative(p)
        if divisor != 0:
            return self.similarity_positive(p) / divisor
        else:
            return 0

    def conservative_similarity(self, p):
        divisor = self.similarity_half_first_positive(p) + self.similarity_negative(p)
        if divisor != 0:
            return self.similarity_half_first_positive(p) / divisor
        else:
            return 0

    def get_feature(self, patch):
        return self.hog.compute(patch)

    def update_positives(self, position):
        feature = self.get_feature(position)
        if feature is not None:
            self.positives.append(feature)

    def update_negatives(self, position):
        feature = self.get_feature(position)
        if feature is not None:
            self.negatives.append(feature)

    def add_new_positive(self, position):
        feature = self.get_feature(position)
        if feature is not None:
            self.new_positives.append(feature)

    def add_new_negative(self, position):
        feature = self.get_feature(position)
        if feature is not None:
            self.new_negatives.append(feature)

    def n_expert(self, n_threshold = 0.2):
        for patch in self.new_positives:
            if self.conservative_similarity(patch) < n_threshold:
                self.update_negatives(patch)
            else:
                self.update_positives(patch)

    def p_expert(self, p_threshold = 0.8):
        for patch in self.new_negatives:
            if self.conservative_similarity(patch) > p_threshold:
                self.update_positives(patch)
            else:
                self.update_negatives(patch)