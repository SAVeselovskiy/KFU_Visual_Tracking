# -*- coding: utf-8 -*-
__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

import cv2
import math
import numpy as np
from CFiles.CHelper import CHelper
import StringIO


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

        self.bbPoints(self.bounding_box)
        self.timer_for_calculate_points = 100
        self.err = None
        self.status = None
        self.simm = None
        self.FB_error = None
        self.fbmed = None

    # возвращает новый boundingBox, новые вычисленные точки, булево значение (True - следит, False - потерял объект)
    def track(self, new_frame, old_position):
        self.bbPoints(self.bounding_box)
        # if self.points is None or self.timer_for_calculate_points <= 0:
        #     self.calculate_points(old_position)
        if self.points is not None:
            new_points, self.status, self.err = cv2.calcOpticalFlowPyrLK(old_position.buffer[0], new_frame, self.points,
                                                                         None, **self.lk_params)
            pointsFB, statusFB, self.FB_error = cv2.calcOpticalFlowPyrLK(new_frame, old_position.buffer[0], new_points,
                                                                         None,
                                                                         **self.lk_params)
            i = 0
            while i < len(self.points):
                self.FB_error[i] = norm(substractPoint(pointsFB[i][0], self.points[i][0]))
                i += 1

            self.normCrossCorrelation(old_position.buffer[0], new_frame, self.points, new_points)
            self.points, new_points, tracked = self.filterPoints(self.points, new_points)

            if not tracked:
                return None

            # новые бокс и точки
            newbox = getNewBB(self.points, new_points, self.bounding_box)
            # if newbox[0] < 0 or newbox[1] < 0 or newbox[0] + newbox[2] > new_frame.shape[0] or newbox[1] + newbox[
            #     3] > new_frame.shape[1]:
            #     return None
            self.points = new_points
            self.bounding_box = newbox
            self.timer_for_calculate_points -= 1
            return self.bounding_box
        else:
            return None

    def calculate_points(self, position):
        if position.is_correct():
            self.timer_for_calculate_points = 100
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
                print "Tracking points count:", len(self.points)
            else:
                print "Can't detect points for Tracking"
                self.points = None
        return self.points

    def bbPoints(self, box):
        max_pts = 10
        stepx = box[2] / max_pts
        stepy = box[3] / max_pts

        points = np.empty(((max_pts - 1) * (max_pts - 1), 1, 2), np.float32)

        y = box[1] + stepy
        x = box[0] + stepx
        i = 0
        while y < box[1] + box[3]:
            x = box[0] + stepx
            while x < box[0] + box[2] and i < len(points):
                points[i][0][0] = x
                points[i][0][1] = y
                x += stepx
                i += 1
            y += stepy
        self.points = points
        return points

    def normCrossCorrelation(self, img1, img2, points1, points2):
        i = 0
        chelper = CHelper()
        while i < len(self.points):
            if self.status[i] == 1:
                rec0 = cv2.getRectSubPix(img1, (10, 10), (points1[i][0][0], points1[i][0][1]))
                rec1 = cv2.getRectSubPix(img2, (10, 10), (points2[i][0][0], points2[i][0][1]))
                res = cv2.matchTemplate(rec0, rec1, eval('cv2.TM_CCOEFF_NORMED'))
                str = np.getbuffer(res)
                io = StringIO.StringIO(str)
                testnum = chelper.getHelpNumber(io.getvalue())
                self.err[i] = testnum
            else:
                self.err[i] = 0.0
            i += 1

    def filterPoints(self, points1, points2):
        self.simm = median(self.err)
        k = 0
        i = 0
        while i < len(points2):
            if self.status[i] == 0:
                i += 1
                continue
            if self.err[i] >= self.simm:
                points1[k][0] = points1[i][0]
                points2[k][0] = points2[i][0]
                self.FB_error[k] = self.FB_error[i]
                k += 1
            i += 1
        if k == 0:
            return None, None, False
        points1 = np.resize(points1, (k, 1, 2))
        points2 = np.resize(points2, (k, 1, 2))
        self.FB_error = np.resize(self.FB_error, (k, 1))

        self.fbmed = median(self.FB_error)
        k = 0
        i = 0
        while i < len(points2):
            # if self.status[i] == 0:
            #     i += 1
            #     continue
            if self.FB_error[i] <= self.fbmed:
                points1[k][0] = points1[i][0]
                points2[k][0] = points2[i][0]
                k += 1
            i += 1
        points1 = np.resize(points1, (k, 1, 2))
        points2 = np.resize(points2, (k, 1, 2))
        if k > 0:
            return points1, points2, True
        return None, None, False
