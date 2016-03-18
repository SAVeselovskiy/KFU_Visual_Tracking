import cv2

image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray,None,fx=0.3,fy=0.3)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
gray = cv2.drawKeypoints(gray,kps,None,(255,0,0),4)
cv2.imshow('SIFT',gray)
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass