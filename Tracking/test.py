from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn import svm, datasets
import numpy as np

ima1 = np.zeros((4,4))
ima2 = np.ones((4,4))
ima1[1:3,1:3] = np.max(ima1[1:3],ima2[1:3])

print ima