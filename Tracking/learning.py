__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import numpy as np

def get_bounding_box(frame, x, y, width, height):
    if x >= 0 and y >= 0 and y+height < frame.shape[0] and x+width < frame.shape[1]:
        return frame[y:y+height, x:x+width]
    else:
        return None

def add_gaussian_noise(bounding_box, mean, sigma):
    if bounding_box is not None:
        return bounding_box + np.random.normal(mean, sigma, bounding_box.shape)
    else:
        return None

class LearningComponent:

    def __init__(self, initial_frame, x, y, width, height, patch_size = (15,15)):
        self.patch_size = patch_size
        self.positives = []
        self.negatives = []
        self.new_positives = []
        self.new_negatives = []
        gray_initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        bounding_box = get_bounding_box(gray_initial_frame, x, y, width, height)
        self.update_positives(bounding_box)
        self.generate_training_examples(gray_initial_frame, x, y, width, height)

    def generate_training_examples(self, gray_initial_frame, x, y, width, height, closest_count = 10, surround_count = 50, small_radius = None, big_radius = None, sigma = 5):
        if small_radius == None:
            small_radius = 0.1 * (width + height) / 2
        if big_radius == None:
            big_radius = (width + height) / 2

        for i in xrange(closest_count):
            fi = i * np.pi / closest_count
            diff_x = x + int(np.cos(fi) * small_radius)
            diff_y = y + int(np.sin(fi) * small_radius)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y, width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

            if bounding_box is not None:
                bounding_box = cv2.resize(bounding_box, None, fx=1.01,fy=1.01)
                generalise = add_gaussian_noise(bounding_box, 0, sigma)
                self.update_negatives(generalise)

                bounding_box = cv2.resize(bounding_box, None, fx=0.99,fy=0.99)
                generalise = add_gaussian_noise(bounding_box, 0, sigma)
                self.update_negatives(generalise)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x-max(int(0.01*width), 1), diff_y, width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x+max(int(0.01*width), 1), diff_y, width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y-max(int(0.01*height), 1), width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y+max(int(0.01*height), 1), width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

        for i in xrange(surround_count):
            fi = i * np.pi / closest_count
            diff_x = x + int(np.cos(fi) * big_radius)
            diff_y = y + int(np.sin(fi) * big_radius)

            bounding_box = get_bounding_box(gray_initial_frame, diff_x, diff_y, width, height)
            generalise = add_gaussian_noise(bounding_box, 0, sigma)
            self.update_negatives(generalise)

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

    def update_positives(self, p):
        if p is not None:
            if p.shape != self.patch_size:
                p = cv2.resize(p, self.patch_size)
                p = np.uint8(np.rint(p))
            self.positives.append(p)

    def update_negatives(self, p):
        if p is not None:
            if p.shape != self.patch_size:
                p = cv2.resize(p, self.patch_size)
                p = np.uint8(np.rint(p))
            self.negatives.append(p)

    def add_new_positive(self, p):
        if p is not None:
            if p.shape != self.patch_size:
                p = cv2.resize(p, self.patch_size)
                p = np.uint8(np.rint(p))
            self.new_positives.append(p)

    def add_new_negative(self, p):
        if p is not None:
            if p.shape != self.patch_size:
                p = cv2.resize(p, self.patch_size)
                p = np.uint8(np.rint(p))
            self.new_negatives.append(p)

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

    def get_init_patch(self):
        if len(self.positives):
            return self.positives[0]
        else:
            return None