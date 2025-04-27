function [H, annotatedFrames] = pos_rppg_ft_corrected(videoSource, isRealTime, l)

faceDetector = vision.CascadeObjectDetector();       
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');

pointTracker = vision.PointTracker('MaxBidirectionalError', 2); % KLT PointTracker which is initialized to tolerate maximum error of 2 pixels

initialized = false; %This is the bool object that allows us to check whether KLT has started or not
foreheadBbox = [0, 0, 0, 0];

videoObject.CurrentTime=0;
frameIndex = 1;
timeSeries_R = [];
timeSeries_G = [];
timeSeries_B = [];

annotatedFrames = {};

%Starting with the initiation of the algorithm based on video source
%(pre-recorded or webcam)
if ~isRealTime
    %Pre-recorded
    if ~exist(videoSource, 'file')
        error('The video file could not been found: %s', videoSource);
    end
    videoObject = VideoReader(videoSource);
    
else
    %Real-time (webcam)
    cam = webcam;
end

%Main loop
while true
    if ~isRealTime
        if ~hasFrame(videoObject), break; end
        frame = readFrame(videoObject);
    else
        frame = snapshot(cam);
    end

    if ~initialized % If this bool object is not yet true, KLT has not started. Thus we need to detect a forehead for the first time to track with KLT
        faceBbox = step(faceDetector, frame);
        
        if ~isempty(faceBbox)
            faceBbox = faceBbox(1,:);
            faceRegion = imcrop(frame, faceBbox);

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
                foreheadY = eyesBbox(1,2) - foreheadHeight - 80;

                foreheadBbox = [foreheadX, foreheadY, foreheadWidth, foreheadHeight];

                if all(foreheadBbox(3:4) > 0)
                    %if size of the foreheadbox is acceptable, KLT is
                    %initialized and the feature points are found in
                    %forehead ROI
                    grayFrame = rgb2gray(frame);
                    points = detectMinEigenFeatures(grayFrame, 'ROI', foreheadBbox);

                    if ~isempty(points)
                        %starting of the KLT
                        initialize(pointTracker, points.Location, frame);
                        initialized = true;
                    end
                end
            end
        end
    else
        %if the KLT is already initialized and working, the feature points
        %are tracked
        [pointsTracked, isFound] = step(pointTracker, frame);
        if sum(isFound) < 3
            initialized = false;
            release(pointTracker);
            continue;
        end
        
        validPoints = pointsTracked(isFound, :);

        %the tracking of the foreheadbox is made by movement estimation
        minX = min(validPoints(:,1));
        minY = min(validPoints(:,2));
        maxX = max(validPoints(:,1));
        maxY = max(validPoints(:,2));
        foreheadBbox = [minX, minY, maxX - minX, maxY - minY];

        foreheadRegion = imcrop(frame, foreheadBbox);
        if size(foreheadRegion,3) == 3
            R = foreheadRegion(:,:,1);
            G = foreheadRegion(:,:,2);
            B = foreheadRegion(:,:,3);
            timeSeries_R(frameIndex) = mean(R(:));
            timeSeries_G(frameIndex) = mean(G(:));
            timeSeries_B(frameIndex) = mean(B(:));
        end
        
        %The visualisation is made based on whether the video is
        %pre-recorded or real time
        if ~isRealTime
            annotatedFrame = insertObjectAnnotation(frame,'rectangle',foreheadBbox, 'Forehead','Color','yellow');
            annotatedFrames{frameIndex} = annotatedFrame;
        else
            % Real-time modda isterseniz anlık gösterim
            imshow(insertObjectAnnotation(frame,'rectangle',foreheadBbox,'Forehead'));
            drawnow;
        end
        
        frameIndex = frameIndex + 1;
        setPoints(pointTracker, validPoints);
    end
    
    %to stop the algorithm at a treshold in real-time
    if isRealTime && frameIndex > 450
        disp('The execution stopped due to being reached to 300 frames.');
        break;
    end
    
end
release(pointTracker);

avgDouble_R = double(timeSeries_R(:));
avgDouble_G = double(timeSeries_G(:));
avgDouble_B = double(timeSeries_B(:));

N = length(avgDouble_R);


H = zeros(1, N);
h_series = zeros(1, N);

S1_series = zeros(1, N);
S2_series = zeros(1, N);


P = [0,  1, -1;
    -2,  1,  1];

C = zeros(3, N);  

for n = 1:N
    %Spatial averaging
    C(:, n) = [avgDouble_R(n); avgDouble_G(n); avgDouble_B(n)];
    
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


if ~isRealTime
    figure('Name','POS-rPPG (Pre-recorded)','NumberTitle','off');
    for i = 1 : length(annotatedFrames)
        subplot(1,2,1);
        imshow(annotatedFrames{i});
        title(sprintf('Frame %d', i));
        
        subplot(1,2,2);
        plot(H(1:i), 'k');
        title('POS-rPPG Pulse Over Time');
        xlabel('Frame Index');
        ylabel('Amplitude');
        grid on;
       
        drawnow;
        pause(0.01); 
    end
else
    figure('Name','POS-rPPG (Real-time Summary)','NumberTitle','off');
    plot(H, 'r');
    title('POS-rPPG Pulse Over Time (Real-time Session)');
    xlabel('Frame Index');
    ylabel('Amplitude');
    grid on;
end

end