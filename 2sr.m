close all; clear all; clc;

videoPath= "C:\Users\demet\OneDrive\Resimler\Film Rulosu\facetrack.mp4";
videoObject = VideoReader(videoPath);%VideoReader, allows us to read an object via a multimedia file

faceDetector = vision.CascadeObjectDetector();%Function that does object detection by using Viola-Johnson algorithm, default:face
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');

videoObject.CurrentTime=0;
frameIndex = 1;
timeSeries_R = {};
timeSeries_G = {};
timeSeries_B = {};

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

                timeSeries_R{frameIndex} = R(:);
                timeSeries_G{frameIndex} = G(:);
                timeSeries_B{frameIndex} = B(:);

                annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', foreheadBbox, 'Forehead');
                annotatedFrames{frameIndex} = annotatedFrame;
            end
            frameIndex = frameIndex + 1;
        end
    end
end

%Unlike other existing algorithms, 2SR does not use spatial average of RGB
%channels over the pixels of ROI. Instead, the mean of V is substracted for
%taking the spatial distribution of skin pixels into account
R_all = cell2mat(timeSeries_R');  
G_all = cell2mat(timeSeries_G');  
B_all = cell2mat(timeSeries_B');  

% We form V as a Nx3 matrix where rows are pixels and the columns are RGB
% channels
V = double([R_all, G_all, B_all]);
%disp(V(1:5, :)); This definition results in 1337592x3 matrix, which
%indicates that we have approximately 13938 pixels within one frame

l = 45; %window size
K = length(timeSeries_R); %number of frames

P = zeros(1, K); %pulse signal is initialized
N = size(V, 1) / K; %number of pixels per frame to extract V_k

SR_all = [];

for k = 1:K 
    startIdx = (k-1) * N + 1;
    endIdx = k * N;
    V_k = V(startIdx:endIdx, :);
    
    C_k = (V_k' * V_k)/N;
    disp(['C_k for frame ', num2str(k), ':']);
    disp(C_k);

    [U_k, Lambda_k] = eig(C_k);  %eigenvalue decomposition
    [eigenvalues, idx] = sort(diag(Lambda_k), 'descend');  %2SR algorithm depends on the assumption of u1 being the principal eigenvector that defines the main direction.
    %Since MATLAB's eig() function does not guarantees the order of
    %eigenvalues and eigenvectors, there is a need for sorting
    U_k = U_k(:, idx);
     if k == 1
        Lambda_tau = eigenvalues;
        U_tau = U_k;
    end

    if (k - l + 1) > 0 %tenporal window of l frames, SR' will be calculated
        tao = k - l + 1;
        
        Lambda_tau = eigenvalues;
        U_tau = U_k;

        SR = [];
        for t = tao:K
            ratio12 = min(max(eigenvalues(1) / Lambda_tau(2), 1e-6), 1e6);
            ratio13 = min(max(eigenvalues(1) / Lambda_tau(3), 1e-6), 1e6);
            SR = [SR; sqrt(ratio12) * (U_k(:, 1)' * U_tau(:, 2)) * U_tau(:, 2)' + sqrt(ratio13) * (U_k(:, 1)' * U_tau(:, 3)) * U_tau(:, 3)'];
        end
       
        SR_all = [SR_all; SR];
    end
    if (k - l + 1) > 0
        SR_window = SR_all(tao:k, :);
       
        SR_1 = SR_window(:, 1);  
        SR_2 = SR_window(:, 2);  
        std_SR1 = std(SR_1);
        std_SR2 = std(SR_2);
        p = SR_1 - (std_SR1 / max(std_SR2, 1e-12)) * SR_2;

        p_centered = p - mean(p);
        
        %overlap adding
        P(k-l+1:k) = P(k-l+1:k) + p_centered';
    end
end

for i = 1 : length(annotatedFrames) %The length of annotatedFrames is choosen for the iteration because the number of frameIndex exceeds this length, 
    % probably because the empty cells where ROI couldn't be detected.
    subplot(1,2,1);
    imshow(annotatedFrames{i});
    pause(0.1);
    title(sprintf('Frame %d', i));

    subplot(1,2,2);
    plot(P(1:i), 'k');
    title('Pulse Over Time');
    xlabel('Frame Index');
    ylabel('Amplitude');
    drawnow;  
     
end



