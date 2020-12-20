import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import bio_lda.lda as lda
from tqdm import tqdm

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)

def generate_dataset(samples=1000, d=50, proportions=[0.5,0.5], separation=0):
    mat = np.random.randn(d,d)
    cov = mat@mat.T

    mean1 = np.random.randn(d)
    mean2 = np.random.randn(d) + separation
    X1 = np.random.multivariate_normal(mean1, cov, int(proportions[0] * samples))
    X2 = np.random.multivariate_normal(mean2, cov, int(proportions[1]  * samples))
    X = np.concatenate((X1,X2))
    y = np.zeros(samples)
    y[int(proportions[0] * samples):] = 1
    idx = np.random.permutation(len(X))
    X, y  = X[idx], y[idx]
    return X, y, mean1, mean2, cov


def scikit_LDA(X,y):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,y)
    y_ = clf.transform(X)
    true_lda_score = max((np.sum(y_[y == 1] > 0) + np.sum(y_[y == 0] < 0)), (np.sum(y_[y == 0] > 0) + np.sum(y_[y == 1] < 0)) )/y.shape[0]
    return y_, true_lda_score, clf


def run_offline(X, y, mean1, mean2, cov, true_lda_score, eta, gamma, epochs=5000 ):
    LDA = lda.Offline_LDA(1, X.shape[1])
    LDA.eta = eta
    LDA.gamma = gamma
    #LDA.w = np.expand_dims(np.linalg.inv(cov)@(mean1-mean2), axis=-1)
    err = []
    metric = []
    optimal = []
    optimal_W = np.linalg.inv(cov)@(mean1-mean2)
    for n_e in tqdm(range(epochs)):
        LDA.fit(mean1, mean2, cov)
        Y = LDA.w.T.dot(X.T)
        err.append(true_lda_score - max((np.sum(Y[:,y == 1] > 0) + np.sum(Y[:,y == 0] < 0)), (np.sum(Y[:,y == 0] > 0) + np.sum(Y[:,y == 1] < 0)) )/X.shape[0])
        metric.append(((LDA.w.T@(mean1-mean2))**2/(LDA.w.T@cov@LDA.w)).item())
        optimal.append(
            ((optimal_W.T@(mean1-mean2))**2/(optimal_W.T@cov@optimal_W)).item() - 
            ((LDA.w.T@(mean1-mean2))**2/(LDA.w.T@cov@LDA.w)).item())
    return LDA, err, metric, optimal

def run_online(X, y,  mean1, mean2, cov_tot,  true_lda_score, epochs=50):
    LDA = lda.Online_LDA(1, X.shape[1])

    err = []
    metric = []
    optimal = []

    for n_e in tqdm(range(epochs)):
        for count, x in enumerate(X):
            if y[count] == 0:
                r = 1
                s = 0
            else:
                r = 0
                s = 1
            LDA.fit_next(x, r, s)
        Y = LDA.w.T.dot(X.T)
        err.append(true_lda_score - max((np.sum(Y[y == 1] > 0) + np.sum(Y[y == 0] < 0)), (np.sum(Y[y == 0] > 0) + np.sum(Y[y == 1] < 0)) )/10000)
        metric.append(((LDA.w.T@(mean1-mean2))**2/(LDA.w.T@cov_tot@LDA.w)).item())
        optimal_W = np.linalg.inv(cov_tot)@(mean1-mean2)
        optimal.append(
            ((optimal_W.T@(mean1-mean2))**2/(optimal_W.T@cov_tot@optimal_W)).item() - 
            ((LDA.w.T@(mean1-mean2))**2/(LDA.w.T@cov_tot@LDA.w)).item())
    return LDA, err, metric, optimal

