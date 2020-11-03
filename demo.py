import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from util import *
from lda import *


# Generate data and labels
X, y = make_classification(n_samples=10000, n_features=500,class_sep=10, n_clusters_per_class=1, n_classes=2)

# Offline Version stats
# Calculate means of each class and total covariance of both classes together
X1 = X[y == 0]
X2 = X[y == 1]
mean1 = X1.mean(0)
mean2 = X2.mean(0)
mean_diff = np.atleast_1d(mean1 - mean2)
cov1 = calculate_covariance_matrix(X1)
cov2 = calculate_covariance_matrix(X2)
cov_tot = cov1 + cov2


# Compute True Accuracy using Sklearn
clf = LinearDiscriminantAnalysis()
clf.fit(X,y)
y_ = clf.transform(X)
true_lda_score = max((np.sum(y_[y == 1] > 0) + np.sum(y_[y == 0] < 0)), (np.sum(y_[y == 0] > 0) + np.sum(y_[y == 1] < 0)) )/10000


# Initialize Online LDA Algorithm
LDA = Online_LDA(1, 500)

n_epoch = 50
err = []

# fit data using algorithm
for n_e in range(n_epoch):
    for count, x in enumerate(X):
        # Set r and s based on label
        if y[count] == 0:
            r = 1
            s = 0
        else:
            r = 0
            s = 1
        LDA.fit_next(x, r, s)
    Y = LDA.w.T.dot(X.T)
    err.append(true_lda_score - max((np.sum(Y[y == 1] > 0) + np.sum(Y[y == 0] < 0)), (np.sum(Y[y == 0] > 0) + np.sum(Y[y == 1] < 0)) )/10000)

# output accuracy and err of model

print(f"classification accuracy of {(true_lda_score - err[-1]) * 100} percent")
plt.scatter(Y,np.zeros(Y.shape), c = y)
plt.show()
plt.plot(err)
plt.show()