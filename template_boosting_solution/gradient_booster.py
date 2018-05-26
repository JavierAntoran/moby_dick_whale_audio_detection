from __future__ import division
import numpy as np
import xgboost as xgb



data = np.load('data/template_features.npy')
labels = np.load('../data/whale_trainlabels.npy')

data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]), order='C')

# train test split
cutoff = np.round(data.shape[0] * 0.1).astype(int)
x_train = data[cutoff:]
t_train = labels[cutoff:]
print('+train = %d' % np.sum(t_train))
x_dev = data[:cutoff]
t_dev = labels[:cutoff]
print('+t_dev = %d' % np.sum(t_dev))

dtrain = xgb.DMatrix(x_train, label=t_train)
dtest = xgb.DMatrix(x_dev, label=t_dev)

param = {}
param['nthread'] = 2
param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

