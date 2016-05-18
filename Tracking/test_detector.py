
# -*- coding: utf-8 -*-

import cv2
import time
from detection import Detector
from learning import LearningComponent
from structure import Position

def get_samples(folder, pos_list_filename, samples_subfolder, max_count):
    result = []
    pos_filenames = folder + pos_list_filename
    with open(pos_filenames) as f:
        files = f.readlines()
    for imfile in files:
        if len(result) >= max_count:
            break
        png_filename = folder + samples_subfolder + imfile.rstrip('\n')
        im = cv2.imread(png_filename)
        result.append(im)
    return result

folder = '/Users/admin/Documents/Магистратура/Грантовый проект Коннова/Dataset/INRIAPerson/'
pos_list = 'train_64x128_H96/pos.lst'
test_post_list = 'test_64x128_H96/pos.lst'
neg_list = 'Train/neg.lst'
samples_subfolder = '96X160H96/Train/'
test_samples_subfolder = '70X134H96/Test/'

init_frames_count = 100
max_train_count = 2*init_frames_count
max_test_count = max_train_count
start_time = time.time()
positive_samples = get_samples(folder, pos_list, samples_subfolder, max_train_count+max_test_count)
negative_samples = get_samples(folder, neg_list, "", 5*max_train_count+max_test_count)
positive_patches = [Position(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 0, frame.shape[1], frame.shape[0]).calculate_patch() for frame in positive_samples]
negative_patches = [Position(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 0, frame.shape[1], frame.shape[0]).calculate_patch() for frame in negative_samples]
print "Positive samples:", len(positive_samples)
print "Negative samples:", len(negative_samples)
print "Generate training set:", time.time() - start_time
print

start_time = time.time()
learning_component = LearningComponent(positive_patches[0])
detector = Detector(learning_component)
for patch in positive_patches[1:max_train_count]:
    learning_component.update_positives(patch)
for patch in negative_patches[:5*max_train_count]:
    learning_component.update_negatives(patch)
print "Update training set:", time.time() - start_time
print

start_time = time.time()
detector.ensemble_classifier.relearn()
print "Learn detector:", time.time() - start_time
print

for param in xrange(10):
    print "Param:", param/10.0
    detector.threshold_patch_variance = 0.3
    detector.threshold_ensemble = param/10.0
    detector.nearest_neighbor_classifier.tetta = 0.4
    start_time = time.time()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for patch in positive_patches[-max_test_count:]:
        if detector.predict_patch(patch) > 0.5:
            tp += 1
        else:
            fn += 1
    for patch in negative_patches[-max_test_count:]:
        if detector.predict_patch(patch) > 0.5:
            fp += 1
        else:
            tn += 1
    print "True positives:", tp
    print "False negatives:", fn
    print "True negatives:", tn
    print "False positives:", fp
    accuracy = 1.0*(tp+tn)/(max_test_count*2)
    print "Accuracy:", accuracy
    precision = 1.0*tp/(tp+fp)
    print "Precision:", precision
    recall = 1.0*tp/(tp+fn)
    print "Recall:", recall
    fscore = 2*precision*recall/(precision+recall)
    print "F-score:", fscore
    print "Testing:", time.time() - start_time
    print