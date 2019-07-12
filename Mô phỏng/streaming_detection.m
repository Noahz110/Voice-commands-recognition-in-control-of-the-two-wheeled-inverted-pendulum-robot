fs = 16e3;
classificationRate = 20;
audioIn = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));

frameLength = frameDuration*fs;
hopLength = hopDuration*fs;
waveBuffer = zeros([fs,1]);

labels = trainedNet.Layers(end).ClassNames;
YBuffer(1:classificationRate/2) = "background";
probBuffer = zeros([numel(labels),classificationRate/2]);

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
addpath(fullfile(matlabroot,'examples','audio','main'))

while ishandle(h)
    
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;
    
    spec = auditorySpectrogram(waveBuffer,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength-hopLength, ...
        'NumBands',numBands, ...
        'Range',[0,8000], ...
        'WindowType','Hann', ...
        'WarpType','Mel', ...
        'SumExponent',2);
    spec = log10(spec + epsil);
    
    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','gpu');
    YBuffer(1:end-1)= YBuffer(2:end);
    YBuffer(end) = YPredicted;
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = probs';
    
    subplot(2,1,1);
    plot(waveBuffer)
    axis tight
    ylim([-0.2,0.2])
    
    subplot(2,1,2)
    pcolor(spec)
    caxis([specMin+2 specMax])
    shading flat
    
    [YMode,count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == YMode,:));
    probThreshold = 0.9;
    subplot(2,1,1);
    if YMode == "background" || count<countThreshold || maxProb < probThreshold
        title(" ")
    else
        title(YMode,'FontSize',20)
    end
    
    drawnow
    
end