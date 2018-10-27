from __future__ import division
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import xgboost as xgb

from sklearn.datasets import make_classification

def shuffle_in_unison(a, b, c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)

# dset, tgets = make_classification(n_samples=200, n_features=5, n_informative=3,
#                            n_classes=2, weights=[.9, .1], shuffle=True)


dset = np.load('data/template_features_combined.npy')
tgets = np.load('data/whale_trainlabels.npy')

shuffle_in_unison(dset, tgets)

x_train, x_test, t_train, t_test = train_test_split(dset, tgets, test_size=0.1, stratify=tgets)


# train test split
# cutoff = np.round(data.shape[0] * 0.1).astype(int)
# x_train = data[cutoff:]
# t_train = labels[cutoff:]
# print('+train = %d' % np.sum(t_train))
# x_dev = data[:cutoff]
# t_dev = labels[:cutoff]
# print('+t_dev = %d' % np.sum(t_dev))

# NOTE: unballanced dataset

# weights = np.zeros(len(t_train))
# weights[t_train==1] = 1
# weights[t_train==0] = 5

 # Define task parameters

param = {}

param['objective'] = 'binary:logistic'
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['silent'] = 1

# Regularization
param['max_depth'] = 3 # of trees #DEFAULT:6
param['gamma'] = 0 # loss reduction to make further tree partition #DEAFULT:0
param['min_child_weight'] = 1 # minimum weight asigned to leaf node in order to continue partition #DEFAULT:1
param['learning_rate'] = 0.3 # (eta) shrinks weights to make learning more conservative  #DEFAULT:0.3

# More Regularization
param['subsample'] = 0.6 # subsample ratio of the training instance to grow trees #DEFAULT:1
param['colsample_bytree'] = 0.9

# Balancing. Use scale_pos_weight OR max_delta_step
ratio = float(np.sum(t_train==0)) / np.sum(t_train==1)
print(ratio)
param['scale_pos_weight'] = ratio
###################
# param['max_delta_step'] = 10 # helps make tree update conservative??  #DEFAULT:0 no limit



dtrain = xgb.DMatrix(x_train, label=t_train)
dtest = xgb.DMatrix(x_test, label=t_test)


evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 100

bst = xgb.train(param, dtrain, num_round, evallist)

y_test_preds = bst.predict(dtest)


fpr, tpr, threshold = metrics.roc_curve(t_test, y_test_preds)

roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

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
plt.savefig('results/best_ROC.png')