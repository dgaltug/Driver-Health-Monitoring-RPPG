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

medianSeries_R = [];
medianSeries_G = [];
medianSeries_B = [];
figure(1);


while hasFrame(videoObject)
    frame = readFrame(videoObject); %Reads the first avialble frame in video file
    faceBbox = step(faceDetector, frame);

    if ~isempty(faceBbox)
        [~, largestIdx] = max(faceBbox(:, 3) .* faceBbox(:, 4)); % area = width * height
        faceBbox = faceBbox(largestIdx, :); %to keep the largest face for processing in case multiple faces are detected
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

                med_R = median(R(:)); 
                med_G = median(G(:)); 
                med_B = median(B(:)); 

                disp(['Average Red: ', num2str(avg_R)]);
                disp(['Average Green: ', num2str(avg_G)]);
                disp(['Average Blue: ', num2str(avg_B)]);

                disp(['Median of Red: ', num2str(med_R)]);
                disp(['Median of Green: ', num2str(med_G)]);
                disp(['Median of Blue: ', num2str(med_B)]);

                timeSeries_R(frameIndex) = avg_R;
                timeSeries_G(frameIndex) = avg_G;
                timeSeries_B(frameIndex) = avg_B;

                medianSeries_R(frameIndex) = med_R;
                medianSeries_G(frameIndex) = med_G;
                medianSeries_B(frameIndex) = med_B;

                
                
                annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', foreheadBbox, 'Forehead');
            end
            frameIndex = frameIndex + 1;
            
        end
        
    end
    %subplot(2, 2, 1);
    imshow(annotatedFrame);
    %pause(1/videoObject.FrameRate);
    pause(0.01);
    
    subplot(2, 2, 2);
    plot(timeSeries_R, 'r');
    hold on;
    plot(medianSeries_R, 'r--');
    title('Red Channel Over Time');
    xlabel('Frame Index');
    ylabel('Average&Median Red Value');
    hold off;
    
    subplot(2, 2, 3);
    plot(timeSeries_G, 'g');
    hold on;
    plot(medianSeries_G, 'g--');
    title('Green Channel Over Time');
    xlabel('Frame Index');
    ylabel('Average&Median Green Value');
    hold off;
    %{
    subplot(2, 2, 4);
    plot(timeSeries_B, 'b');
    hold on;
    plot(medianSeries_B, 'b--');
    title('Blue Channel Over Time');
    xlabel('Frame Index');
    ylabel('Average&Madian Blue Value');
    hold off;
    %}
end

