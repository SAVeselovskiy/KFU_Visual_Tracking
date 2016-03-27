# -*- coding: utf-8 -*-
__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import math
import numpy as np

def substractPoint(point1, point2):
        return (point2[0] - point1[0], point2[1] - point1[1])

def median(A):  # A cheat to find the median in our base case
    T = list(A)  # copies list!
    T.sort()
    i = len(T) / 2  # in list of 4 selects the 3rd element
    return T[int(i)]

def norm(point):
    result = math.sqrt(point[0] * point[0] + point[1] * point[1])
    return result

def getNewBB(points1, points2, oldBB):
    npoints = len(points1)
    xoff = np.empty([npoints, 1], dtype=np.float32)
    yoff = np.empty([npoints, 1], dtype=np.float32)
    for i in range(0, npoints):
        xoff[i] = points2[i][0][0] - points1[i][0][0]
        yoff[i] = points2[i][0][1] - points1[i][0][1]
    dx = median(xoff)
    dy = median(yoff)
    s = 1.0
    if npoints > 1:
        d = []
        for i in range(0, npoints):
            for j in range(i + 1, npoints):
                first = substractPoint(points2[i][0], points2[j][0])
                second = substractPoint(points1[i][0], points1[j][0])
                normfirst = norm(first)
                normsecond = norm(second)
                if normsecond == 0:
                    normsecond = 1
                res = normfirst / normsecond
                d.append(res)
        s = median(d)
        # if s > 1.1 or s < 0.95:
        #     s = (s + 1)/2
    # print 's = ', s
    s = 1.0
    s1 = 0.5 * (s - 1) * oldBB[2]
    s2 = 0.5 * (s - 1) * oldBB[3]
    temp = oldBB[0] + dx - s1
    x = math.floor(temp) if round(temp * 10 % 10) < 5 else math.ceil(temp)
    temp = oldBB[1] + dy - s2
    y = math.floor(temp) if round(temp * 10 % 10) < 5 else math.ceil(temp)
    temp = oldBB[2] * s
    w = math.floor(temp) if round(temp * 10 % 10) < 5 else math.ceil(temp)
    temp = oldBB[3] * s
    h = math.floor(temp) if round(temp * 10 % 10) < 5 else math.ceil(temp)
    newBB = (int(x), int(y), int(w), int(h))
    return newBB

class Tracker:
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, initial_position):
        self.bounding_box = (initial_position.x, initial_position.y, initial_position.width, initial_position.height)
        self.bb = initial_position.get_bounding_box()

        self.detector = cv2.xfeatures2d.SURF_create(5000)
        self.calculate_points(initial_position)

    # возвращает новый boundingBox, новые вычисленные точки, булево значение (True - следит, False - потерял объект)
    def track(self, new_frame, old_position):
        if self.points is None:
            self.calculate_points(old_position)
        if self.points is not None:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_position.frame, new_frame, self.points, None, **self.lk_params)
            good_new = new_points[st == 1]
            if len(good_new) == 0:
                # последние правильные бокс и точки
                return None
            # новые бокс и точки
            newbox = getNewBB(self.points, new_points, self.bounding_box)
            self.points = new_points
            self.bounding_box = newbox
            return self.bounding_box
        else:
            return None

    def calculate_points(self, position):
        kp, des = self.detector.detectAndCompute(position.get_bounding_box(), None)
        if len(kp) > 0:
            p = kp[0].pt
            arr = np.array([[[p[0] + position.x, p[1] + position.y]]], np.float32)
            i = 1
            while i < len(kp):
                b = np.array([[[kp[i].pt[0] + position.x, kp[i].pt[1] + position.y]]], np.float32)
                arr = np.append(arr, b, 0)
                i += 1
            self.points = arr
        else:
            self.points = None
        return self.points