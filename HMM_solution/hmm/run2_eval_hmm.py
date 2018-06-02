from __future__ import division
from modules.HMM import *
from modules.utils import *

from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)

mkdir('results')

basePath = '../data/'
data = np.load(basePath + 'processed_data_swt_dct.npy')
labels = np.load(basePath + 'whale_trainlabels.npy')

print('data shape:', data.shape)
print('data type:', type(data))

# normalize data
data_mean = np.mean(data, axis=(0, 1), keepdims=True)
data -= data_mean
data_std = np.std(data, axis=(0, 1), keepdims=True)
data /= data_std
#
# Positive HMM
Q = 15 # N states
G = np.empty((Q), dtype=object)
for q in range(Q):
    G[q] = gmm_EM(nb_clust=3, dim=30, centers=None, covars=None, weights=None)
hmm_pos = HMM(Q, G, p0=None, debug=False)
hmm_pos.load('hmm_pos')
#
# Negative HMM
Q = 15  # N states
G = np.empty((Q), dtype=object)
for q in range(Q):
    G[q] = gmm_EM(nb_clust=3, dim=30, centers=None, covars=None, weights=None)
hmm_neg = HMM(Q, G, p0=None, debug=False)
hmm_neg.load('hmm_neg')

data = [a for a in data]

pos_trace = []
neg_trace = []
pos_log_like = np.zeros(len(data))
neg_log_like = np.zeros(len(data))
p = np.zeros((len(data), 2))

for i, d in enumerate(data):
    print('running pass %d of %d' % (i, len(data)))
    pos_trace_, pos_log_like[i] = hmm_pos.eval(d)
    neg_trace_, neg_log_like[i] = hmm_neg.eval(d)
    pos_trace.append(pos_trace_)
    neg_trace.append(neg_trace_)


pos_log_like -= pos_log_like.mean()
pos_log_like /= pos_log_like.std()

neg_log_like -= neg_log_like.mean()
neg_log_like /= neg_log_like.std()

pos_log_like = np.expand_dims(pos_log_like, axis=1)
neg_log_like = np.expand_dims(neg_log_like, axis=1)

out_loglikes = np.concatenate((pos_log_like, neg_log_like), axis=1)
p = softmax(out_loglikes, dim=1)

p = p[:, 0]

# print(labels)
# print(p)
#
# print(pos_log_like[:100])
# print(neg_log_like[:100])


fpr, tpr, threshold = metrics.roc_curve(labels, p)
roc_auc = metrics.auc(fpr, tpr)

# SAVE ROC CURVE
plt.figure()
lw = 1.2
plt.plot(fpr, tpr, color='C0',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='C2', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Best Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('results/2HMM_ROC.png')