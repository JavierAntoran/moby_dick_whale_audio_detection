from __future__ import division
import numpy as np
from scipy import signal
import pywt
from scipy.fftpack import dct

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

def windower(x, M, N):
    # M avance entre vetanas
    # N windowsize

    T   = x.shape[0]
    m   = np.arange(0, T-N+1, M) # comienzos de ventana
    L   = m.shape[0] # N ventanas
    ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1,L)) + np.ones((N,1)) * m
    X   = x[ind.astype(int)]
    return X.transpose()


def gen_whalefb_mtx(NFFT, nfilt, sample_rate, low_freq_mel=None):
    if low_freq_mel is None:
        low_freq_mel = 40

    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale

    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

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

wavelet = 'db2'
level=3
aprox_idx = level - np.array([1,2,3])

#downsample in freq
downsample_factor = 2
fs = int(2000 / downsample_factor)
NFFT = int(512 / downsample_factor)

mfb = gen_whalefb_mtx(NFFT=NFFT, nfilt=32, sample_rate=fs, low_freq_mel=50)

N = int(fs * 0.25)
M = int(fs * 0.011)

W = np.expand_dims(np.hamming(N), axis=0)

mfb_starts = np.array([2, 3, 2])
mfb_ends = np.array([29, 24, 14])

num_cepst = np.array([13, 12, 5])

# Returns 160x30
def get_wfb_swt_dct(x):
    out = np.zeros(np.sum(num_cepst))

    # returns a tuple of size=level of tuples of size 2 (ac, dc)
    ac_dc = pywt.swt(x, wavelet, level, start_level=0, axis=-1)
    ac_dc = np.asarray(ac_dc)
    approx = ac_dc[aprox_idx, 0]

    approx = signal.decimate(approx, downsample_factor, axis=1)

    for i in range(approx.shape[0]):

        a_win = windower(approx[i], M, N)

        a_hamm = a_win * W
        sa = np.abs(np.fft.rfft(a_hamm, n=NFFT, axis=1))
        a_mfb = np.matmul(sa, mfb.T)

        a_mfb = a_mfb[:, mfb_starts[i]:mfb_ends[i]]

        a_mfb = dct(a_mfb, type=2, axis=1, norm='ortho')[:, 0:num_cepst[i]]

        if i == 0:
            out = a_mfb
        else:
            out = np.concatenate((out, a_mfb), axis=1)

    return out

#####################

ready_data = np.zeros((sounds.shape[0], 160, 30))  #sum(num_cepst)=30

for i in range(sounds.shape[0]):
    x = sounds[i]
    ready_data[i] = get_wfb_swt_dct(x)


print(ready_data.shape)
print(ready_data.dtype)
# ready_data = ready_data.astype(np.float32)
print(ready_data.dtype)

np.save('data/processed_data_swt_dct.npy', ready_data)