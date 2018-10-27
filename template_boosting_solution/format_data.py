import numpy as np
from scipy.signal import decimate

def windower(x, M, N):
    # M avance entre vetanas
    # N windowsize

    T   = x.shape[0]
    m   = np.arange(0, T-N+1, M) # comienzos de ventana
    L   = m.shape[0] # N ventanas
    ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1,L)) + np.ones((N,1)) * m
    X   = x[ind.astype(int)]
    return X.transpose()

labels = np.load('data/whale_trainlabels.npy')
sounds = np.load('data/whale_traindata.npy')

decimate_factor = 2
fs = 2000 / decimate_factor

NFFT = 256

N = int(fs * 0.25)
M = int(fs * 0.011)

W = np.expand_dims(np.hamming(N), axis=0)

ready_data = np.zeros((sounds.shape[0], 160, int(NFFT / 2)))

for i in range(sounds.shape[0]):
    x = sounds[i]
    x = decimate(x, decimate_factor)

    x_win = windower(x, M, N)
    x_hamm = x_win * W

    s = np.abs(np.fft.rfft(x_hamm, n=NFFT, axis=1))
    s = s[:, 1:]  # eliminate DC

    ready_data[i] = s

    ready_data -= ready_data.mean(axis=(1, 2), keepdims=True)
    ready_data /= ready_data.std(axis=(1, 2), keepdims=True)

print(ready_data.shape)
print(ready_data.dtype)
# ready_data = ready_data.astype(np.float32)
print(ready_data.dtype)

np.save('data/processed_data_spectrum_250.npy', ready_data)