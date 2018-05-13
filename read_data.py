import numpy as np
from numpy import genfromtxt
import soundfile as sf
import os

data = np.zeros((30000, 2000 * 2))
i = 0

dir = 'whale_data/train'

for i in range(1,30001):
    filename = 'train' + str(i) + '.aiff'
    print(dir + "/" + filename)
    x, fs = sf.read(dir + "/" + filename)
    data[i-1, :] = x


np.save('whale_traindata.npy', data)
#

labels_file = 'whale_data/train.csv'
label_data = genfromtxt(labels_file, dtype=None, delimiter=',', skip_header=1, usecols=1)
print(label_data.shape)
np.save('whale_trainlabels.npy', label_data)
