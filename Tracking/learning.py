__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np

def add_gaussian_noise(bounding_box, mean, sigma):
    if bounding_box is not None:
        return bounding_box + np.random.normal(mean, sigma, bounding_box.shape)
    else:
        return None

class LearningComponent:

    def __init__(self, init_patch):
        self.positives = []
        self.negatives = []
        self.new_positives = []
        self.new_negatives = []
        self.new_samples_count = 0
        # winSize = (16,16)
        # blockSize = (4,4)
        # blockStride = (4,4)
        # cellSize = (4,4)
        # nbins = 9
        # derivAperture = 1
        # winSigma = 4.
        # histogramNormType = 0
        # L2HysThreshold = 2.0000000000000001e-01
        # gammaCorrection = 1
        # nlevels = 64
        # self.descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        self.descriptor = cv2.HOGDescriptor()
        self.update_positives(init_patch)
        self.init_patch = init_patch

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
        if len(self.negatives) != 0 and len(self.positives) != 0:
            samples = []
            positive_weight = 1.0*len(self.negatives)/(len(self.negatives)+len(self.positives))
            negative_weight = 1.0*len(self.positives)/(len(self.negatives)+len(self.positives))
            weights = np.append(positive_weight*np.ones(len(self.positives)),negative_weight*np.ones(len(self.negatives)))
            targets = np.append(np.ones(len(self.positives)),np.zeros(len(self.negatives)))
            for positive in self.positives:
                samples.append(positive.calculate_feature(self.descriptor))
            for negative in self.negatives:
                samples.append(negative.calculate_feature(self.descriptor))
            return samples, weights, targets
        else:
            return np.array([]), np.array([]), np.array([])

    def NCC(self, pi, pj):
        # pi, pj - patches
        CV_TM_CCOEFF_NORMED = 5
        try:
            x = cv2.matchTemplate(pi.small_content, pj.small_content, CV_TM_CCOEFF_NORMED)[0][0]
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

    def update_positives(self, patch):
        if patch != None:
            patch.calculate_feature(self.descriptor)
            self.positives.append(patch)
            self.new_samples_count += 1

    def update_negatives(self, patch):
        if patch != None:
            patch.calculate_feature(self.descriptor)
            self.negatives.append(patch)
            self.new_samples_count += 1

    def add_new_positive(self, patch):
        if patch != None:
            self.new_positives.append(patch)

    def add_new_negative(self, patch):
        if patch != None:
            self.new_negatives.append(patch)

    def n_expert(self, n_threshold = 0.2):
        for patch in self.new_positives:
            if self.conservative_similarity(patch) < n_threshold:
                self.update_negatives(patch)
            else:
                self.update_positives(patch)
        self.new_positives = []

    def p_expert(self, p_threshold = 0.8):
        for patch in self.new_negatives:
            if self.conservative_similarity(patch) > p_threshold:
                self.update_positives(patch)
            else:
                self.update_negatives(patch)
        self.new_negatives = []