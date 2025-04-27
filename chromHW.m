close all; clear all; clc;

videoPath= "C:\Users\demet\OneDrive\Resimler\Film Rulosu\facetrack.mp4";
videoObject = VideoReader(videoPath);%VideoReader, allows us to read an object via a multimedia file

faceDetector = vision.CascadeObjectDetector();%Function that does object detection by using Viola-Johnson algorithm, default:face
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');

videoObject.CurrentTime=0;
frameIndex = 1;
timeSeries_R = [];
timeSeries_G = [];
timeSeries_B = [];

annotatedFrames = {};

figure(1);


while hasFrame(videoObject)
    frame = readFrame(videoObject); %Reads the first avialble frame in video file
    faceBbox = step(faceDetector, frame);

    if ~isempty(faceBbox)
        annotatedFrame = insertObjectAnnotation(frame, 'rectangle', faceBbox, 'Face');%Syntax RGB=iOA(variable,shape,position,label)
        title('Detected Face');
        faceRegion = imcrop(frame, faceBbox(1, :)); %focus on the detected face for further detecting nose and eyes

        eyesBbox = step(eyeDetector, faceRegion); %The variable storing the matrix of bounding boxes detected by the eye detector
        if ~isempty(eyesBbox) %Check if eyes are detected
            eyesBbox(:, 1:2) = eyesBbox(:, 1:2) + faceBbox(1, 1:2);
        end

        noseBbox= step(noseDetector,faceRegion);
        if ~isempty(noseBbox) %Check if the nose is detected
            noseBbox(:, 1:2) = noseBbox(:, 1:2) + faceBbox(1, 1:2);
        end

        if ~isempty(eyesBbox) && ~isempty(noseBbox)
            eyesCenterX = eyesBbox(1, 1) + eyesBbox(1, 3) / 2;
            eyesCenterY = eyesBbox(1, 2) + eyesBbox(1, 4) / 2;

            foreheadWidth = eyesBbox(1, 3) * 0.4;
            foreheadHeight = eyesBbox(1, 4) * 1.0;
            foreheadX = eyesCenterX - foreheadWidth / 2;
            foreheadY = eyesBbox(1,2) - foreheadHeight - 70;

            foreheadBbox = [foreheadX, foreheadY, foreheadWidth, foreheadHeight];
            if all(foreheadBbox(3:4) > 0)
                foreheadRegion = imcrop(frame, foreheadBbox); %extracts the pixel values from the forehead region defined by foreheadBbox

                R = foreheadRegion(:,:,1); % Red channel
                G = foreheadRegion(:,:,2); % Green channel
                B = foreheadRegion(:,:,3); % Blue channel

                avg_R = mean(R(:)); % Average Red channel value
                avg_G = mean(G(:)); % Average Green channel value
                avg_B = mean(B(:)); % Average Blue channel value

                timeSeries_R(frameIndex) = avg_R;
                timeSeries_G(frameIndex) = avg_G;
                timeSeries_B(frameIndex) = avg_B;

                annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', foreheadBbox, 'Forehead');
                annotatedFrames{frameIndex} = annotatedFrame;
            end
            frameIndex = frameIndex + 1;
        end
    end
end

%After the R-G-B datas are obtained as a data which is average of all the
%pixels within ROI, normalization and remaining steps take place outside
%the loop
avgDouble_R = double(timeSeries_R(:));
avgDouble_G = double(timeSeries_G(:));
avgDouble_B = double(timeSeries_B(:));

%Windowing for improving robustness against artifacts
l = 45; %window size=window duration x FPS, FPS:22.5, window duration is selected as 2 seconds since it must be between 1.5 and 3 for capturing heart rate.                 
overlap = 22.5; %overlap between windows with the size of %50   
hannWin = hann(l);
N = length(avgDouble_R);

fs = videoObject.FrameRate;
filterOrder = 4;
[b, a] = butter(4, [0.7, 4]/(fs/2), 'bandpass');

pulseChrom = zeros(1, N);

for n = l:overlap:N
    %current window is called
    window_R = avgDouble_R(n-l+1:n);
    window_G = avgDouble_G(n-l+1:n);
    window_B = avgDouble_B(n-l+1:n);

    R_windowed = window_R .* hannWin';
    G_windowed = window_G .* hannWin';
    B_windowed = window_B .* hannWin';

    if (length(window_R) >= 2 * filterOrder)
        %Normalization
        normRGB = sqrt(window_R.^2 + window_G.^2 + window_B.^2);
        
        smallIdx = (normRGB < 1e-12); %This is where the algorithm will detect the frames that has sqrt value near zero, and the relevant data in the array will be assigned to zero
        
        R_norm(~smallIdx) = window_R(~smallIdx)./normRGB(~smallIdx);
        G_norm(~smallIdx) = window_G(~smallIdx)./normRGB(~smallIdx);
        B_norm(~smallIdx) = window_B(~smallIdx)./normRGB(~smallIdx);
        
        R_norm(smallIdx) = 0;
        G_norm(smallIdx) = 0;
        B_norm(smallIdx) = 0;
        
        %Chrominance Signals
        X = 3.*R_norm - 2.0 .* G_norm;
        Y = 1.5 .* R_norm + G_norm - 1.5 .* B_norm;
        
        
        %Applying band-pass filter to X and Y
        X_filt = filtfilt(b, a, X);
        Y_filt = filtfilt(b, a, Y);
        
        % alpha-scaling
        stdX = std(X_filt);
        stdY = std(Y_filt);
        alpha = stdX / max(stdY, 1e-12);
        s = X_filt - alpha .* Y_filt;
        
        % overlap-adding
        s = s - mean(s);
        disp(['s: ', num2str(s)])
        pulseChrom(n-l+1:n) = pulseChrom(n-l+1:n) + s;
    end
end 


for i = 1 : length(annotatedFrames) %The length of annotatedFrames is choosen for the iteration because the number of frameIndex exceeds this length, 
    % probably because the empty cells where ROI couldn't be detected.
    subplot(1,2,1);
    imshow(annotatedFrames{i});
    pause(0.1);
    title(sprintf('Frame %d', i));

    subplot(1,2,2);
    plot(pulseChrom(1:i), 'k');
    title('PulseChrom Over Time');
    xlabel('Frame Index');
    ylabel('Amplitude');
    %axis([0 frameIndex min(pulseChrom) max(pulseChrom)]);
    drawnow;  
     
end

