__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2

class Position:
    def __init__(self, buffer, x, y, width, height):
        self.buffer = buffer
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.patch = None

    def update(self, x=None, y=None, width=None, height=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        self.patch = None

    def clean_frame(self):
        self.buffer[0] = None
        self.patch = None

    def is_correct(self):
        return self.x >= 0 and self.y >= 0 and self.width > 0 and self.height > 0 and self.x + self.width <= self.buffer[0].shape[1] and self.y + self.height <= self.buffer[0].shape[0]

    def point_left_up(self):
        return (self.x, self.y)

    def point_right_down(self):
        return (self.x + self.width, self.y + self.height)

    def get_bounding_box(self):
        if self.is_correct():
            return self.buffer[0][self.y:self.y+self.height, self.x:self.x+self.width]
        else:
            return None

    def calculate_patch(self):
        if self.patch is not None:
            return self.patch
        else:
            bounding_box = self.get_bounding_box()
            if bounding_box is not None:
                return Patch(bounding_box)
            else:
                return None

    def get_window(self):
        return (self.x, self.y, self.width, self.height)


class Patch:
    def __init__(self, bounding_box):
        self.content = cv2.resize(bounding_box, (64,128))
        self.small_content = cv2.resize(self.content, (16,16))
        self.feature = None

    def calculate_feature(self, descriptor):
        # return self.content.ravel()
        if self.feature is None:
            self.feature = descriptor.compute(self.content).ravel()
        return self.feature
