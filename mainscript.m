clear all; close all; clc;
[H, annotatedFrames] = pos_rppg_ft_corrected("C:\Users\demet\OneDrive\Resimler\Film Rulosu\facetrack.mp4", false, 45);

fs = 30;
rp = 20; %the attenuation value to surpress ripples in the stopband (dB)
lowCut  = 0.8;
highCut = 2.0;

[b,a] = butter(6, [lowCut highCut] / (fs/2), 'bandpass');

H_filt = filtfilt(b, a, H);

%{
figure;
plot(H_filt, 'r');
title('POS-rPPG Butterworth Filtered Pulse Over Time');
xlabel('Frame Index');
ylabel('Amplitude');


[b,a] = cheby2(4, rp, [lowCut highCut] / (fs/2), 'bandpass');

H_filt = filtfilt(b, a, H);

figure;
plot(H_filt, 'r');
title('POS-rPPG Chebyshev Filtered Pulse Over Time');
xlabel('Frame Index');
ylabel('Amplitude');
%}


fs = 30;  

N = length(H_filt);             
t = (0:N-1)/fs;                 

H_hat = fft(H_filt);

PSD=H_hat.*conj(H_hat)/N;                 
P1 = PSD(1:floor(N/2)+1);        

f = fs*(0:floor(N/2))/N;        

%step to find the peak frequency
[~, peakIndex] = max(P1);       
peakFreq = f(peakIndex);

%Hertz to BPM translation
BPM = peakFreq * 60;            


figure('Name','FFT of Filtered RPPG','NumberTitle','off');
subplot(2,1,1);
plot(t, H_filt, 'k');
xlabel('Time (s)'); ylabel('Amplitude');
title('Butterworth Filtered RPPG Signal in Time Domain');
grid on;

subplot(2,1,2);
plot(f, P1, 'b', 'LineWidth',1.2);
plot(f, P1, 'b', 'LineWidth',1.2);
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('Single-Sided FFT');
grid on;
hold on;
plot(peakFreq, P1(peakIndex), 'ro');  
legend('FFT Spectrum','Dominant Peak');

disp(['BPM Assumption = ', num2str(BPM)]);

