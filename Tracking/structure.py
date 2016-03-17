__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

class Position:
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

    def get_bounding_box(self):
        if self.x >= 0 and self.y >= 0 and self.y+self.height < self.frame.shape[0] and self.x+self.width < self.frame.shape[1]:
            return self.frame[self.y:self.y+self.height, self.x:self.x+self.width]
        else:
            return None