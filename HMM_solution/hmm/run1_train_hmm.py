from __future__ import division
from modules.HMM import *

basePath = '../data/'
data = np.load(basePath + 'processed_data_swt_dct.npy')
labels = np.load(basePath + 'whale_trainlabels.npy')

data_pos = data[labels == 1]
data_neg = data[labels == 0]

print('data_pos shape:', data_pos.shape)
print('data_pos type:', type(data_pos))

print('data_neg shape:', data_neg.shape)
print('data_neg type:', type(data_neg))

# normalize data
data_mean = np.mean(data, axis=(0, 1), keepdims=True)
data -= data_mean
data_std = np.std(data, axis=(0, 1), keepdims=True)

data_pos -= data_mean
data_pos /= data_std

data_neg -= data_mean
data_neg /= data_std

Q = 10  # N states

G = np.empty((Q), dtype=object)
for q in range(Q):
    G[q] = gmm_EM(nb_clust=3, dim=30, centers=None, covars=None, weights=None)

hmm_pos = HMM(Q, G, p0=None, debug=False)
hmm_pos.train([a for a in data_pos], iterations=20, N_only_gmm=2)
hmm_pos.save('hmm_pos')

G = np.empty((Q), dtype=object)
for q in range(Q):
    G[q] = gmm_EM(nb_clust=3, dim=30, centers=None, covars=None, weights=None)

hmm_neg = HMM(Q, G, p0=None, debug=False)
hmm_neg.train([a for a in data_neg], iterations=20, N_only_gmm=2)
hmm_neg.save('hmm_neg')