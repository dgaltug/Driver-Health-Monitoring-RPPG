function [H, annotatedFrames] = rppg(videoSource, isRealTime, l)

faceDetector = vision.CascadeObjectDetector();       
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');

pointTracker = vision.PointTracker('MaxBidirectionalError', 2); % KLT PointTracker which is initialized to tolerate maximum error of 2 pixels

initialized   = false; %This is the bool object that allows us to check whether KLT has started or not
foreheadBbox  = [0 0 0 0];
foreW = []; foreH = []; %initialized forehead size

frameIndex    = 1;
timeSeries_R  = [];
timeSeries_G  = [];
timeSeries_B  = [];
annotatedFrames = {};



%Starting with the initiation of the algorithm based on video source
%(pre-recorded or webcam)
if ~isRealTime
    %Pre-recorded
    if ~exist(videoSource,'file'), error('File not found: %s',videoSource); end
    videoObject = VideoReader(videoSource);
else
    %Real-time (webcam)
    cam = webcam;
    fs      = 30;
    winLen  = 3*fs; %these are defined to shape a sliding window for real-time process
    buf_R   = nan(1,winLen);%buffers for all RGB channels are set to store data from 3 second intervals
    buf_G   = nan(1,winLen);
    buf_B   = nan(1,winLen);
    bufPtr  = 0; 
    lastBPM = NaN;
    bufSize   = 10;              % 10 BPM = 30 seconds
    bpmBuf    = NaN(1, bufSize); % running buffer
    bpmCount  = 0;               % amount of collected BPM
    muPrev       = NaN; 
    sdPrev = NaN; % statistics from previous window
    stableStreak = 0;
    calibDone    = false;
    meanStable   = NaN; sdStable = NaN;

    fig = figure('Name','Real-Time rPPG','NumberTitle','off');

    axCam = axes(fig,'Position',[0.05 0.30 0.9 0.65]);   % real-time video figure
    axPDF = axes(fig,'Position',[0.10 0.05 0.8 0.2]);    % Gaussian
    axPDF.Visible = 'on';   % grid vs.
    hold(axPDF,'on');
    xlabel(axPDF,'BPM');  ylabel(axPDF,'Probablity Density Function');
end

%% Main Loop %%
while true
    if ~isRealTime
        if ~hasFrame(videoObject), break; end
        frame = readFrame(videoObject);
    else
        frame = snapshot(cam);
    end
    
    %% ROI Initialization
    if ~initialized  % If this bool object is not yet true, KLT has not started. Thus we need to detect a forehead for the first time to track with KLT
        faceBbox = step(faceDetector, frame);
        if ~isempty(faceBbox)
            faceBbox = faceBbox(1,:);
            faceRegion = imcrop(frame,faceBbox);

            eyesBbox = step(eyeDetector,faceRegion); %The variable storing the matrix of bounding boxes detected by the eye detector
            if ~isempty(eyesBbox) %Check if eyes are detected
                eyesBbox(:,1:2)=eyesBbox(:,1:2)+faceBbox(1,1:2); 
            end

            noseBbox = step(noseDetector,faceRegion);
            if ~isempty(noseBbox) %Check if the nose is detected
                noseBbox(:,1:2)=noseBbox(:,1:2)+faceBbox(1,1:2); 
            end

            if ~isempty(eyesBbox) && ~isempty(noseBbox)
                eyesCenterX = eyesBbox(1,1)+eyesBbox(1,3)/2;
                eyesCenterY = eyesBbox(1,2)+eyesBbox(1,4)/2;

                foreheadWidth  = eyesBbox(1,3)*0.4;
                foreheadHeight = eyesBbox(1,4)*1.0;
                foreheadX = eyesCenterX - foreheadWidth/2;
                foreheadY = eyesBbox(1,2) - foreheadHeight - 80;

                foreheadBbox = [foreheadX foreheadY foreheadWidth foreheadHeight];

                if all(foreheadBbox(3:4)>0)
                    %if size of the foreheadbox is acceptable, KLT is
                    %initialized and the feature points are found in
                    %forehead ROI
                    gray = rgb2gray(frame);
                    points = detectMinEigenFeatures(gray,'ROI',foreheadBbox);

                    if ~isempty(points)
                        %starting of the KLT
                        initialize(pointTracker,points.Location,frame);
                        initialized = true;
                        foreW = foreheadBbox(3); %obtained height and width of the ROI bounding box  
                        foreH = foreheadBbox(4);
                    end
                end
            end
        end

    else  
        %% ROI Tracking
        %if the KLT is already initialized and working, the feature points
        %are tracked
        [pointsTracked, isFound] = step(pointTracker,frame);
        if sum(isFound) < 3
            initialized = false; 
            release(pointTracker); 
            continue;
        end
        validPoints = pointsTracked(isFound,:);

        
        % PCA-based angle definition
        center = mean(validPoints,1);
        Ccov   = cov(validPoints); %covariance matrix
        [V,D]  = eig(Ccov); %eigendecomposition
        [~,id] = max(diag(D));
        vec    = V(:,id);
        theta_raw  = atan2(vec(2),vec(1)); %angle (radians)

        % The corners of the bounding box is defined at center (0,0) initially
        corners0 = [-foreW/2 -foreH/2;
                      foreW/2 -foreH/2;
                      foreW/2  foreH/2;
                     -foreW/2  foreH/2];
        R = [cos(theta_raw) -sin(theta_raw); sin(theta_raw) cos(theta_raw)]; %rotation matrix
        rotCorners = (R*corners0')' + center;  %it rotates and carries the bounding box to the center

        % 3) Mask & RGB average
        mask = poly2mask(double(rotCorners(:,1)),double(rotCorners(:,2)),size(frame,1),size(frame,2));
        Rchan = frame(:,:,1); 
        Gchan=frame(:,:,2); 
        Bchan=frame(:,:,3);
        timeSeries_R(frameIndex) = mean(Rchan(mask));
        timeSeries_G(frameIndex) = mean(Gchan(mask));
        timeSeries_B(frameIndex) = mean(Bchan(mask));

        if isRealTime
            bufPtr = mod(bufPtr, winLen) + 1;
            buf_R(bufPtr) = timeSeries_R(frameIndex);
            buf_G(bufPtr) = timeSeries_G(frameIndex);
            buf_B(bufPtr) = timeSeries_B(frameIndex);
        end
        
        %% Visualisation
        %Rather than a rectangular shape, polygon is used with insertShape
        %function to define a more flexible bounding box
        poly = rotCorners'; 
        poly = poly(:)';   % [x1 y1 … x4 y4]
        if ~isRealTime
            annotatedFrame = insertShape(frame,'Polygon',poly,'Color','red','LineWidth',4);
            annotatedFrames{frameIndex} = annotatedFrame;
        else
            vis = insertShape(frame,'Polygon',poly,'Color','red','LineWidth',4);
            imshow(vis,'Parent',axCam);
            if ~isnan(lastBPM), title(sprintf('Real-Time BPM ≈ %.1f',lastBPM)); end
            drawnow;
        end

        frameIndex = frameIndex + 1;
        setPoints(pointTracker,validPoints);
    end

    %% Real-time POS and BPM estimation
    %3 second windowing for real time BPM update
    if  isRealTime && frameIndex>winLen && mod(frameIndex,30)==0
        idx = mod(bufPtr:bufPtr+winLen-1,winLen)+1;
        seg_R = double(buf_R(idx)); 
        seg_G = double(buf_G(idx)); 
        seg_B = double(buf_B(idx));

        %POS algorithm step
        H_win = zeros(1, winLen);  P = [0 1 -1; -2 1 1];
        C = [seg_R; seg_G; seg_B];
        for n = l:winLen
            if (n - l + 1) > 0
                m = n-l+1;
                Cmean = mean(C(:,m:n),2);
                Cnorm = C(:,m:n)./Cmean;
                S = P*Cnorm;
                alpha = std(S(1,:))/max(std(S(2,:)),1e-12);
                h = S(1,:)+alpha*S(2,:);
                H_win(m:n) = H_win(m:n)+(h-mean(h));
            end
        end

        % Butterworth 4. order 0.8–2 Hz
        [b,a] = butter(4,[0.8 2]/(fs/2),'bandpass');
        Hf = filtfilt(b,a,H_win);

        % FFT ve BPM
        Nw = length(Hf);
        t = (0:Nw-1)/fs;
        Hhat = fft(Hf);

        PSD=Hhat.*conj(Hhat)/Nw;
        P1 = PSD(1:floor(Nw/2)+1);
        f  = fs*(0:floor(Nw/2))/Nw;
       
        [~,idxPk]=max(P1);
        if idxPk>1 && idxPk<length(P1)
            a0=P1(idxPk-1); 
            a1=P1(idxPk); 
            a2=P1(idxPk+1);
            delta=0.5*(a0-a2)/(a0-2*a1+a2);
            peakF=(idxPk-1+delta)*fs/winLen;
        else 
            peakF=f(idxPk); 
        end
        lastBPM=peakF*60;
        
        if isRealTime
            bpmCount = bpmCount + 1;
            idx      = mod(bpmCount-1, bufSize) + 1;   % 1..bufSize index
            bpmBuf(idx) = lastBPM;                     % writing the new BPM into the buffer
            
            if mod(bpmCount, bufSize) == 0
                muCur = mean(bpmBuf);
                sdCur = std(bpmBuf);

                if ~calibDone
                    %stability test
                    if bpmCount >= 2*bufSize           % at least 2 windows
                        if abs(muCur-muPrev) < 1 && abs(sdCur-sdPrev) < .5
                            stableStreak = stableStreak + 1;
                        else
                            stableStreak = 0;
                        end
                    end
    
                    % case 1: min 20 BPM + 3 consecutive stable windows
                    if bpmCount >= 20 && stableStreak >= 3
                        calibDone  = true;
                        meanStable = muCur;  
                        sdStable = sdCur;
                        fprintf('[CALIB] stable: μ=%.1f  σ=%.1f\n',muCur,sdCur);
                    end
                    
                    % case 2: fail-safe 30 BPM (about 90 seconds)
                    if ~calibDone && bpmCount >= 30
                        calibDone  = true;
                        meanStable = muCur;  
                        sdStable = sdCur;
                        fprintf('[CALIB] 90 s done: μ=%.1f  σ=%.1f\n',muCur,sdCur);
                    end
                end
                muPrev = muCur;  
                sdPrev = sdCur; %comparison with next window

                cla(axPDF); %clear the old one
                x = linspace(muCur-4*sdCur, muCur+4*sdCur, 201);
                pdf = 1/(sdCur*sqrt(2*pi)) * exp(-(x-muCur).^2/(2*sdCur^2));
                plot(axPDF,x,pdf,'b','LineWidth',1.5);
                title(axPDF,sprintf('\\mu=%.1f  \\sigma=%.1f',muCur,sdCur));
                grid(axPDF,'on');
            elseif calibDone && ishandle(axPDF)
                axPDF.Color = [0.94 0.94 0.94]; 
                hold(axPDF,'off');
            end 

            if calibDone                      
                lowThr  = meanStable - sdStable;      
                highThr = meanStable + sdStable;
            
                if lastBPM < lowThr || lastBPM > highThr
                    fprintf('[ALERT] BPM %.1f  (Limit: %.1f-%.1f)\n',lastBPM, lowThr, highThr);
                    
                    showAlert(fig, lastBPM);     
                else
                    
                    delete(findall(fig,'Tag','alertBox'));
                end
           end
            
        end

    end
    
    %to stop the algorithm at a threshold in real-time
    if isRealTime && frameIndex>2700
        disp('The execution stopped after 2700 frames.'); break;
    end
end

release(pointTracker);

%% Pre-recorded POS 
if ~isRealTime
    avgDouble_R = double(timeSeries_R(:));
    avgDouble_G = double(timeSeries_G(:));
    avgDouble_B = double(timeSeries_B(:));
    
    N = length(avgDouble_R);
    
    
    H = zeros(1, N);
    
    
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
    H = [];  annotatedFrames = [];
    
end

function showAlert(fig, bpmValue)
    
    old = findall(fig,'Tag','alertBox');
    if ~isempty(old)
        delete(old); 
    end

    annotation(fig,'textbox', [0 0.94 1 0.05], 'String', sprintf('ALERT  %.0f  BPM', bpmValue), 'EdgeColor','none', 'HorizontalAlignment','center', 'Color','r', 'FontSize',18, 'FontWeight','bold', 'Tag','alertBox');
end
end