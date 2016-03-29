from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn import svm, datasets
import numpy as np

print np.count_nonzero([1,0,2,3,0,0])