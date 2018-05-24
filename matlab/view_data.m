%%
clc
close all
clear all
%
Nsample = 6;

%audiofilename = sprintf('../whale_data/train/train%d.aiff', Nsample);
audiofilename = sprintf('../../whale_data/right_whale/train%d.aiff', Nsample);
%audiofilename = sprintf('../../whale_data/no_right_whale/train%d.aiff', Nsample);

[x,Fs] = audioread(audiofilename);

downsample_factor = 2; % go to 1kHz
x = downsample(x, downsample_factor);
Fs = Fs/downsample_factor;

% csvfile = '../whale_data/train.csv';
% labels = csvread(csvfile, 1, 1);
% pos_labels = find(labels == 1);
% label = labels(Nsample);

% fprintf('sample %d, label %d\n', Nsample, label)
%%

soundsc(x,Fs)

NFFT = 128;

M = Fs * 0.01; %avance entre vetanas
N = Fs * 0.03; %windowsize

X = windower(x, M, N);%
% M avance entre vetanas
% N windowsize
%
W = hamming(N);
X_hamm = W .* X; % hamming window, automatic broadcasting
%
% fft default applies to first dimension -> columns
S = abs(fft(X_hamm, NFFT));
S = S(1:end/2,:);
%

f = linspace(0, Fs/2, NFFT);
nwins = 1:size(S,2);

figure
imagesc(nwins, f, S)
axis xy
ylabel('freq (Hz)')
xlabel('Nwindow')
title('Spectogram S')


%% delta features

d1 = [-1.0000,   -0.7500,   -0.5000,   -0.2500,         0,    0.2500,    0.5000,   0.7500,    1.0000];
d2 = [ 1.0000,    0.2500,   -0.2857,   -0.6071,   -0.7143,   -0.6071,   -0.2857,   0.2500,    1.0000];
S_d  = filter2(d1,S);
S_dd = filter2(d2,S);

figure
imagesc(nwins, f, S_d)
axis xy
ylabel('freq (Hz)')
xlabel('Nwindow')
title('delta Spectogram S')
