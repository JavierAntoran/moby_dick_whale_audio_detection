clc
clear all
close all
%% define stats
Fs = 2000 / 2;
Nfilt = 32;
NFFT = 256;

%% regular MFB
mfb_mtx = f_banco_filtros_mel(NFFT/2, Nfilt, Fs);

ff = linspace(0, Fs/2, NFFT/2);

figure
imagesc([], ff, mfb_mtx)
xlabel('Nwindow')
ylabel('frequency')
axis xy

%% Whale filter bank

wfb_mtx = f_banco_filtros_balleno(NFFT/2, Nfilt, Fs);

ff = linspace(0, Fs/2, NFFT/2);

figure
imagesc([], ff, wfb_mtx)
xlabel('Nwindow')
ylabel('frequency')
axis xy
figure
plot(wfb_mtx)