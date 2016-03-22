import cv2

frame = cv2.imread('test2.jpg')
hog = cv2.HOGDescriptor()
x = hog.compute(frame)

pass