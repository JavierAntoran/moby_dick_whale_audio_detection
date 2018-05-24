%% Load data
clc
close all
clear all
%
Nsample = 13;

audiofilename = sprintf('../whale_data/train/train%d.aiff', Nsample);

[x,Fs] = audioread(audiofilename);

downsample_factor = 1; % go to 1kHz
x = downsample(x, downsample_factor);
Fs = Fs/downsample_factor;

csvfile = '../whale_data/train.csv';
labels = csvread(csvfile, 1, 1);
pos_labels = find(labels == 1);
label = labels(Nsample);

fprintf('sample %d, label %d\n', Nsample, label)


%% Stationary SWT
close all
level = 3;
wtype = 'sym8';
[swa,swd] = swt(x, level, wtype);

%% stationary time signals
t = (1:length(x))/Fs;

figure
ax(1) = subplot(level+1, 1, 1);
plot(t, x);
grid on
axis tight
xlabel('time (s)')
title('original signal')
for i = 2:level+1
    
    ax(i) = subplot(level+1, 1, i);
    plot(t, swa(i-1, :));
    grid on
    axis tight
    xlabel('time (s)')
    title(sprintf('Aproximation signal N= %d', i-1))
    
    
end
linkaxes(ax, 'x')

%% Stationary multilevel approximation coefficients
NFFT = 512;

M = Fs * 0.01; %avance entre vetanas
N = Fs * 0.03; %windowsize

W = hamming(N);

for i = 1:level
    
    X = windower(swa(i,:), M, N);%
    
    X_hamm = W .* X; 
    S = abs(fft(X_hamm, NFFT));
    S = S(1:end/2,:);
    
    f = linspace(0, Fs/2, NFFT);
    nwins = 1:size(S,2);

    figure
    imagesc(nwins, f, S)
    axis xy
    ylabel('freq (Hz)')
    xlabel('Nwindow')
    title(sprintf('Spectrogram swa%d, label: %d', i, label)) 
end

%% Stationary detail coefficients
NFFT = 512;

M = Fs * 0.01; %avance entre vetanas
N = Fs * 0.25; %windowsize

W = hamming(N);

for i = 1:level
    
    X = windower(swd(i,:), M, N);%
    
    X_hamm = W .* X; 
    S = abs(fft(X_hamm, NFFT));
    S = S(1:end/2,:);
    
    f = linspace(0, Fs/2, NFFT);
    nwins = 1:size(S,2);

    figure
    imagesc(nwins, f, S)
    axis xy
    ylabel('freq (Hz)')
    xlabel('Nwindow')
    title(sprintf('Spectrogram swd%d, label: %d', i, label)) 
end

%% Discrete multiresolution DWT
clc
close all
levels = 4;
wtype = 'db2';

[c, l] = wavedec(x, levels, wtype);

getlevel = 3;

dwa = appcoef(c, l, wtype, getlevel);
dwd = detcoef(c, l, getlevel);

%% Get approximation and detail spectrograms
Fsn = Fs/getlevel;
%
NFFT = 256;
M = round(Fsn * 0.01); %avance entre vetanas
N = round(Fsn * 0.03); %windowsize
a = windower(dwa, M, N);%
d = windower(dwd, M, N);%
W = hamming(N);
%
a_hamm = W .* a; 
Sa = abs(fft(a_hamm, NFFT));
Sa = Sa(1:end/2,:);
% f = linspace(0, Fsn/2, NFFT);
% nwins = 1:size(Sa,2);
%
d_hamm = W .* d; 
Sd = abs(fft(d_hamm, NFFT));
Sd = Sd(1:end/2,:);
f = linspace(0, Fsn/2, NFFT);
nwins = 1:size(Sd,2);
%
figure
imagesc(nwins, f, Sa)
axis xy
ylabel('freq (Hz)')
xlabel('Nwindow')
title(sprintf('Spectrogram dwt approx %d, label: %d', i, label)) 

figure
imagesc(nwins, f, Sd)
axis xy
ylabel('freq (Hz)')
xlabel('Nwindow')
title(sprintf('Spectrogram dwt details %d, label: %d', i, label)) 

