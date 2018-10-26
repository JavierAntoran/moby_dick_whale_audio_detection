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

filter1 = np.array([-1.0000, -0.7500, -0.5000, -0.2500, 0, 0.2500, 0.5000, 0.7500, 1.0000])
filter2 = np.array([1.0000, 0.2500, -0.2857, -0.6071, -0.7143, -0.6071, -0.2857, 0.2500, 1.0000])

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

def get_mtx_deltas(X, filter1, filter2):
    Nwin = X.shape[0]
    W1 = get_delta_mtx(filter1, Nwin)
    W2 = get_delta_mtx(filter2, Nwin)
    delta1 = np.dot(W1, X)
    delta2 = np.dot(W2, X)

    # add new dimension to get delta features as channels
    out = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(delta1, axis=2), np.expand_dims(delta2, axis=2)),
                         axis=2)
    return out


def get_delta_mtx(the_filter, Nwin):
    # returns Nwin by Nwin matrix W1. W1 * base_features = delta1
    Fsize = len(the_filter)
    T_pad = (Fsize - 1)
    Nwin_pad = Nwin + T_pad
    pad_s = int(T_pad / 2)
    pad_e = int(T_pad / 2)  # + Nwin_pad%2
    # print('padding start, end:', pad_s, pad_e)
    # Generate Filter
    W1 = np.zeros((Nwin, Nwin_pad))
    for i in range(Nwin_pad - Fsize + 1):
        W1[i, i:i + Fsize] = the_filter
    W1 = W1[:, pad_s:]
    W1 = W1[:, :-pad_e]
    return W1


mkdir('data')

decimate_factor = 2
fs = 2000 / decimate_factor

NFFT = 64

N = int(fs * 0.025)
M = int(fs * 0.01)

W = np.expand_dims(np.hamming(N), axis=0)

ready_data = np.zeros((sounds.shape[0], 198, int(NFFT / 2), 3))

for i in range(sounds.shape[0]):
    x = sounds[i]
    x = decimate(x, decimate_factor)

    x_win = windower(x, M, N)
    x_hamm = x_win * W

    s = np.abs(np.fft.rfft(x_hamm, n=NFFT, axis=1))
    s = s[:, 1:]  # eliminate DC

    s_augment = get_mtx_deltas(s, filter1, filter2)
    #     s_augment = np.rollaxis(s_augment, 2, 0)

    ready_data[i] = s_augment

print(ready_data.shape)
print(ready_data.dtype)
# ready_data = ready_data.astype(np.float32)
print(ready_data.dtype)

np.save('data/processed_data.npy', ready_data)