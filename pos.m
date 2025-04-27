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

%Similar to CHROM algorithm, C(t) signal is averaged and normalization
%applied to C(t). However, the normalization process is slightly different
%this time. Ban
avgDouble_R = double(timeSeries_R(:));
avgDouble_G = double(timeSeries_G(:));
avgDouble_B = double(timeSeries_B(:));

%Windowing for improving robustness against artifacts
l = 45; %window size=window duration x FPS, FPS:22.5, window duration is selected as 2 seconds since it must be between 1.5 and 3 for capturing heart rate.                 
%overlap = 22.5; %overlap between windows with the size of %50           
N = length(avgDouble_R);

H = zeros(1,N);
h_series = zeros(1, N);

P = [0,  1, -1;
    -2,  1,  1];

%For being able to perform standard deviation, we must store the S matrix that is produced for every temporal window 
S1_series = zeros(1, N);  % Store S1 for all frames
S2_series = zeros(1, N);  % Store S2 for all frames


for n = 1:N
    %Spatial averaging
    C(:,n) = [avgDouble_R(n); avgDouble_G(n); avgDouble_B(n)]; %mean of RGB values over the pixels of ROI  for a single frame
    
    if (n - l + 1) > 0
        m = n - l + 1;

        % Temporal normalization
        C_mean = mean(C(:, m:n), 2);
        C_normalized = C(:, m:n) ./ C_mean;
        
        % Projection onto orthogonal plane
        S = P * C_normalized;


        % alpha tuning with projected signals this time
        std_S1 = std(S(1,:));
        std_S2 = std(S(2,:));
        alpha = std_S1 / max(std_S2, 1e-12);
        h = S(1,:) + alpha * S(2,:);
        
        meanh = mean(h);
        h_c = h - meanh;
        
        % Overlap-adding
        H(m:n) = H(m:n) + h_c;
        
    end
end


for i = 1 : length(annotatedFrames) %The length of annotatedFrames is choosen for the iteration because the number of frameIndex exceeds this length, 
    % probably because the empty cells where ROI couldn't be detected.
    subplot(1,2,1);
    imshow(annotatedFrames{i});
    pause(0.01);
    title(sprintf('Frame %d', i));

    subplot(1,2,2);
    plot(H(1:i), 'k');
    title('Pulse Over Time');
    xlabel('Frame Index');
    ylabel('Amplitude');
    drawnow;  
     
end

