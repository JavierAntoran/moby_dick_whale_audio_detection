from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.draw import line, polygon
from scipy import ndimage
import time
import sys
import cPickle
from skimage.feature import hog

def image_statistics(Z):
    #Input: Z, a 2D array, hopefully containing some sort of peak
    #Output: cx,cy,sx,sy,skx,sky,kx,ky
    #cx and cy are the coordinates of the centroid
    #sx and sy are the stardard deviation in the x and y directions
    #skx and sky are the skewness in the x and y directions
    #kx and ky are the Kurtosis in the x and y directions
    #Note: this is not the excess kurtosis. For a normal distribution
    #you expect the kurtosis will be 3.0. Just subtract 3 to get the
    #excess kurtosis.
    import numpy as np

    h,w = np.shape(Z)

    x = range(w)
    y = range(h)


    #calculate projections along the x and y axes
    yp = np.sum(Z,axis=1)
    xp = np.sum(Z,axis=0)

    #centroid
    cx = np.sum(x*xp)/(np.sum(xp))
    cy = np.sum(y*yp)/np.sum(yp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)

    return cx,cy,sx,sy,skx,sky,kx,ky


# Asume 160 x 128 image
chunk=int(sys.argv[1])


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

spectrograms = np.load('data/processed_data_norm_spectrum_250_%d.npy' % (chunk,))


print('spectrograms loaded and normalized, shape:', spectrograms.shape)
# spec = spectograms[7]
# xcorr = signal.correlate2d(spec, templates[Ntemplate], mode='full', boundary='fill', fillvalue=0)
#
# plt.figure()
# plt.imshow(xcorr.T)
# plt.gca().invert_yaxis()
# plt.title('Cross correlation of spectogram with template %d' % Ntemplate)
# plt.savefig('xcorr_%d.png' % Ntemplate)
# plt.show()

def save(where, what):
    file = open(where, 'wb')
    file.write(cPickle.dumps(what))
    file.close()

features = np.zeros((spectrograms.shape[0], len(templates), 11))

for i in range(spectrograms.shape[0]):
    tic0 = time.time()
    for temp_idx in range(len(templates)):

        # print('template %d of %d for spectrogram %d of %d' % (temp_idx, len(templates), i, spectograms.shape[0]))

        spec = spectrograms[i]
        xcorr = signal.correlate2d(spec, templates[temp_idx], mode='full', boundary='fill', fillvalue=0)
        xcorr += xcorr.min()

        xcorr_max = xcorr.max()
        xcorr_mean = xcorr.mean()
        xcorr_sdt = xcorr.std()

        cx, cy, sx, sy, skx, sky, kx, ky = image_statistics(xcorr)
        features[i, temp_idx, :] = np.array([xcorr_max, xcorr_mean, xcorr_sdt, cx, cy, sx, sy, skx, sky, kx, ky])
        
    tic1 = time.time()
    print('finished spectrogram %d of %d. Ellapsed time: %d s' % (i, spectrograms.shape[0], tic1-tic0))
# out = hog(xcorr, block_norm='L2-Hys')

print(features.shape)

np.save('data/template_features_%d.npy'%(chunk,), features)
