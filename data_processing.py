from __future__ import division
from scipy import signal
import pywt
import sys
import concurrent.futures
import numpy as np
import time

data = np.load('./data/whale_traindata.npy')
labels = np.load('./data/whale_trainlabels.npy')
indexes = np.arange(0,data.shape[0])


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
    
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    return fbank


def get_wfb_swt(x):
    
    out = np.zeros(60)

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
      
      if i == 0:
          out = a_mfb
      else:
          out = np.concatenate((out, a_mfb), axis=1)
      
    return out

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



data_proc = np.zeros((data.shape[0],160, 60))

start = time.time()

args = [np.array(a) for a in data[:,:]]

arg_indexes = np.arange(0,len(args))

po = len(arg_indexes)
poi = 1
data_proc = np.zeros((data.shape[0],160, 60))
with concurrent.futures.ProcessPoolExecutor() as executor:
  for index, result in zip(arg_indexes, executor.map(get_wfb_swt, args, chunksize=500)):
    print("Processed %d (%d left)" % (index, po-poi))
    data_proc[index,:,:] = result
    poi = poi + 1
end = time.time()
print("Concurrent elapsed time: %f" % (end - start,))

np.save('./data/concurrent_data_proc.npy', data_proc)
