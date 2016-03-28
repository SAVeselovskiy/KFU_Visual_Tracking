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
        self.tracked_patches = []

    def get_single_window(self, position, detected_windows, tracked_window, correl_track_detect = 0.75, detect_threshold = 0.8):
        single_window = None
        is_visible = False
        if tracked_window is None:
            self.tracked_patches = []
            if len(detected_windows) != 0:
                # max_similarity = 0
                # for detected_window, patch in detected_windows:
                #     similarity = self.learning_component.conservative_similarity(patch)
                #     if similarity > max_similarity:
                #         max_similarity = similarity
                #         single_window = detected_window
                #         is_visible = True
                max_probability = 0
                for detected_window, patch in detected_windows:
                    probability = (patch)
                    if probability > max_probability:
                        max_probability = probability
                if max_probability > detect_threshold:
                    single_window = detected_window
                    is_visible = True
                    for window, patch in detected_windows:
                        if window != single_window:
                            self.learning_component.update_negatives(patch)
                        else:
                            self.learning_component.update_positives(patch)
        elif len(detected_windows) == 0:
            single_window = tracked_window
            self.tracked_patches.append(position.calculate_patch())
        else:
            max_fscore = 0
            for detected_window, patch in detected_windows:
                intersection = windows_intersection(detected_window, tracked_window)
                precision = 1.0*intersection/(detected_window[2]*detected_window[3])
                recall = 1.0*intersection/(tracked_window[2]*tracked_window[3])
                if precision + recall != 0:
                    fscore = 2*precision*recall/(precision+recall)
                    if fscore > max_fscore:
                        max_fscore = fscore
                        max_detected_window = detected_window
            if max_fscore > correl_track_detect:
                single_window = max_detected_window
                is_visible = True
                for patch in self.tracked_patches:
                    self.learning_component.update_positives(patch)
                for window, patch in detected_windows:
                    if window != single_window:
                        self.learning_component.update_negatives(patch)
                    else:
                        self.learning_component.update_positives(patch)
        return single_window, is_visible