from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.draw import line, polygon
import time
from skimage.feature import hog

# Asume 160 x 128 image

length = np.arange(5, 101, 5)  # 11ms advance -> 55 to 1100ms lengths
height = np.arange(1, 76, 5)  # 3.9Hz per coefficient 0 to 1111ms lengths

print('len length', len(length))
print('len height', len(height))
print('total N templates' ,len(length)*len(height))

templates = []

#size added to edges of template
addsize = 6

for h in height:
    for l in length:

        tm = -1 * np.ones((l+addsize, h+addsize))
        startpoint = np.array([int(addsize/2), int(addsize/2)])
        endpoint = startpoint + np.array([l, h])

        rr, cc = line(startpoint[0], startpoint[1], endpoint[0]-1, endpoint[1]-1)
        tm[rr, cc] = 1
        templates.append(tm)


# Ntemplate = 200

# plt.figure()
# plt.imshow(templates[Ntemplate].T)
# plt.gca().invert_yaxis()
# plt.title('Template %d' % Ntemplate)
# plt.savefig('Template_%d.png' % Ntemplate)
# plt.show()

spectograms = np.load('data/processed_data_spectrum_250.npy')


spectograms -= spectograms.mean(axis=(1,2), keepdims=True)
spectograms /= spectograms.std(axis=(1,2), keepdims=True)


print('spectograms loaded and normalized, shape:', spectograms.shape)
# spec = spectograms[7]
# xcorr = signal.correlate2d(spec, templates[Ntemplate], mode='full', boundary='fill', fillvalue=0)
#
# plt.figure()
# plt.imshow(xcorr.T)
# plt.gca().invert_yaxis()
# plt.title('Cross correlation of spectogram with template %d' % Ntemplate)
# plt.savefig('xcorr_%d.png' % Ntemplate)
# plt.show()



features = np.zeros((spectograms.shape[0], len(templates), 3))

for i in range(spectograms.shape[0]):
    tic0 = time.time()
    for temp_idx in range(len(templates)):

        # print('template %d of %d for spectrogram %d of %d' % (temp_idx, len(templates), i, spectograms.shape[0]))

        spec = spectograms[i]
        xcorr = signal.correlate2d(spec, templates[temp_idx], mode='full', boundary='fill', fillvalue=0)

        xcorr_max = xcorr.max()
        xcorr_mean = xcorr.mean()
        xcorr_sdt = xcorr.std()

        features[i, temp_idx, :] = np.array([xcorr_max, xcorr_mean, xcorr_sdt])

    tic1 = time.time()
    print('finished spectogram %d of %d. Ellapsed time: %d s' % (i, spectograms.shape[0], tic1-tic0))
# out = hog(xcorr, block_norm='L2-Hys')

print(features.shape)

np.save('data/template_features.npy', features)