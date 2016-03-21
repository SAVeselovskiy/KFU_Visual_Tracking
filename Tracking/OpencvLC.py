__author__ = 'sergejveselovskij'
import numpy as np
import cv2
import learning

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7,
                      useHarrisDetector=False)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21),
                 maxLevel=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(5000)
kp, des = surf.detectAndCompute(old_frame, None)
while not len(kp) > 0:
    ret, old_frame = cap.read()
    kp, des = surf.detectAndCompute(old_frame, None)

p = kp[0].pt
arr = np.array([[[p[1], p[0]]]], np.float32)
i = 1
while i < len(kp):
    b = np.array([[[kp[i].pt[1], kp[i].pt[0]]]], np.float32)
    arr = np.append(arr, b, 0)
    i += 1
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while (1):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, arr, None, **lk_params)
    #
    # # Select good points
    good_new = p1[st == 1]

    # # draw the tracks
    for i, (new) in enumerate(good_new):
        a, b = new.ravel()
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # frame = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

        # Now update the previous frame and previous points
    old_frame = frame.copy()
    arr = p1

cv2.destroyAllWindows()
cap.release()
