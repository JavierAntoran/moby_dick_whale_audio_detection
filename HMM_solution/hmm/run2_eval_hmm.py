from __future__ import division
from modules.HMM import *
import pickle

basePath = '../data/'
data = np.load(basePath + 'processed_data_swt_dct.npy')
labels = np.load(basePath + 'whale_trainlabels.npy')
data = data[labels==1]

print('data shape:', data.shape)
print('data type:', type(data))

# normalize data
data_mean = np.mean(data, axis=2, keepdims=True)
data -= data_mean
data_std = np.std(data, axis=2, keepdims=True)
data /= data_std

Q = 10  # N states
G = np.empty((Q), dtype=object)
for q in range(Q):
    G[q] = gmm_EM(nb_clust=3, dim=30, centers=None, covars=None, weights=None)

hmm = HMM(Q, G, p0=None, debug=False)


hmm.load('my_hmm_model')

print(hmm.A)
print(hmm.G[0].weights)
# data_use = data
# hmm.train([a for a in data_use[:100]], iterations=10, N_only_gmm=2)

