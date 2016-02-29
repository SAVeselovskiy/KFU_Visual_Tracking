__author__ = 'IVMIT KFU: Gataullin Ravil & Veselovkiy Sergei'

def windows_intersection(window1, window2):
    left = max(window1[0], window2[0])
    right = min(window1[0]+window1[2], window2[0]+window2[2])
    up = max(window1[1], window2[1])
    down = min(window1[1]+window1[3], window2[1]+window2[3])
    if right > left and down > up:
        return (right - left) * (down - up)
    else:
        return 0

class Integrator:
    def get_single_window(self, gray_frame, detected_windows, tracked_window, learning_component):
        single_window = None
        max_similarity = 0
        for detected_window, patch in detected_windows:
            if tracked_window == None or windows_intersection(detected_window, tracked_window) > 0:
                # x, y, width, height = detected_window
                # bounding_box = gray_frame[y:y+height, x:x+width]
                # patch = cv2.resize(bounding_box, (15,15))
                similarity = learning_component.conservative_similarity(patch)
                if similarity > max_similarity:
                    max_similarity = similarity
                    single_window = detected_window
        return single_window
