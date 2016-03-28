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
    def __init__(self, learning_component):
        self.learning_component = learning_component
        self.timer_for_detector = 100

    def get_single_window(self, position, detected_windows, tracked_window):
        single_window = None
        is_visible = False
        if len(detected_windows) == 0:
            single_window = tracked_window
        elif tracked_window is not None and self.timer_for_detector > 0:
            if len(detected_windows) > 0:
                max_fscore = 0
                for detected_window, patch in detected_windows:
                    intersection = windows_intersection(detected_window, tracked_window)
                    precision = 1.0*intersection/(detected_window[2]*detected_window[3])
                    recall = 1.0*intersection/(tracked_window[2]*tracked_window[3])
                    if precision + recall != 0:
                        fscore = 2*precision*recall/(precision+recall)
                        if fscore > max_fscore:
                            max_fscore = fscore
                            single_window = detected_window
                # print "F-score:", max_fscore
                if max_fscore > 0.8:
                    single_window = detected_window
                    is_visible = True
            self.timer_for_detector -= 1
        else:
            max_similarity = 0
            for detected_window, patch in detected_windows:
                similarity = self.learning_component.conservative_similarity(patch)
                if similarity > max_similarity:
                    max_similarity = similarity
                    single_window = detected_window
                    is_visible = True
            self.timer_for_detector = 100

        if single_window is not None:
            if single_window != tracked_window:
                position.update(position.frame, *single_window)
            if tracked_window is not None:
                self.learning_component.update_positives(position.calculate_patch())
                for window, patch in detected_windows:
                    if window != single_window:
                        self.learning_component.update_negatives(patch)
            else:
                self.learning_component.add_new_positive(position.calculate_patch())
                for window, patch in detected_windows:
                    if window != single_window:
                        self.learning_component.add_new_negative(patch)
        else:
            for window, patch in detected_windows:
                self.learning_component.add_new_negative(patch)
        return single_window, is_visible