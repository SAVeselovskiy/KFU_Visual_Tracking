import numpy as np
import time
import cv2

def run_main():
    cap = cv2.VideoCapture(0)
    # Read the first frame of the video
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    faceCascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
    recognised = False
    while not recognised:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) != 0:
            recognised = True
            c,r,w,h = faces[0]

    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle of the building that we're tracking
    # c,r,w,h = 900,650,70,70
    track_window = (c,r,w,h)

    # Create mask and normalized histogram
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
    i = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        extension = 200
        track_rect = frame[y-extension:y+h+extension, x-extension:x+w+extension]
        # track_rect = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(track_rect, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            recognised = False
            while not recognised:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame = cv2.flip(frame,1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) != 0:
                    recognised = True
                    x,y,w,h = faces[0]

                i+=1
                print 'Search faces: %s' % i
                cv2.putText(frame, 'Search faces...', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                cv2.imshow('Tracking', frame)
            track_window = (x,y,w,h)

            # Create mask and normalized histogram
            roi = frame[r:r+h, c:c+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv2.putText(frame, 'Tracked', (x-25,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        i+=1
        print 'Tracking: %s' % i
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    run_main()