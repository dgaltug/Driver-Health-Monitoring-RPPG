clear all; close all; clc;

videoPath = "C:\\Users\\demet\\OneDrive\\Resimler\\Film Rulosu\\facetilting.mp4";
videoObject = VideoReader(videoPath);

% Initialize Face Detector and Point Tracker
faceDetector = vision.CascadeObjectDetector();
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

videoObject.CurrentTime = 0;
frameIndex = 1;
timeSeries_R = [];
timeSeries_G = [];
timeSeries_B = [];

annotatedFrames = {};

initialized = false;
foreheadBbox = [0, 0, 0, 0]; % Initialize foreheadBbox

figure(1);
while hasFrame(videoObject)
    frame = readFrame(videoObject);
    if ~initialized
        faceBbox = step(faceDetector, frame);
        if ~isempty(faceBbox)
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
                foreheadY = eyesBbox(1,2) - foreheadHeight - 80;

                foreheadBbox = [foreheadX, foreheadY, foreheadWidth, foreheadHeight];
            end
            points = detectMinEigenFeatures(rgb2gray(frame), 'ROI', foreheadBbox);
            initialize(pointTracker, points.Location, frame);
            initialized = true;
        end
    else
        [points, validity] = step(pointTracker, frame);
        if sum(validity) < 3
            initialized = false;
            release(pointTracker);
            continue;
        end
        % Track forehead region
        minX = min(points(validity, 1));
        minY = min(points(validity, 2));
        maxX = max(points(validity, 1));
        maxY = max(points(validity, 2));
        foreheadBbox = [minX, minY, maxX - minX, maxY - minY];
    end

    if all(foreheadBbox(3:4) > 0)
        foreheadRegion = imcrop(frame, foreheadBbox);
        avg_R = mean(foreheadRegion(:,:,1), 'all');
        avg_G = mean(foreheadRegion(:,:,2), 'all');
        avg_B = mean(foreheadRegion(:,:,3), 'all');

        timeSeries_R(frameIndex) = avg_R;
        timeSeries_G(frameIndex) = avg_G;
        timeSeries_B(frameIndex) = avg_B;
    end
    annotatedFrame = insertObjectAnnotation(frame, 'rectangle', foreheadBbox, 'Forehead', 'Color', 'yellow');
    annotatedFrames{frameIndex} = annotatedFrame;
    imshow(annotatedFrame);
    title(['Frame ', num2str(frameIndex)]);
    drawnow;
    pause(0.01);

    frameIndex = frameIndex + 1;
    
end
release(pointTracker);

