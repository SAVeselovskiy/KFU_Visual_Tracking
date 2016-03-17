__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

class CurrentPosition:
    def __init__(self, frame, x, y, width, height):
        self.frame = frame
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def update(self, frame, x, y, width, height):
        self.frame = frame
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def point_left_up(self):
        return (self.x, self.y)

    def point_right_down(self):
        return (self.x + self.width, self.y + self.height)
