function Xbkg = backgroundSpectrograms(ads,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing background spectrograms...");

logVolumeRange = log10(volumeRange);
numBkgFiles = numel(ads.Files);       
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));

numHops = segmentDuration/hopDuration - 2;
Xbkg = zeros(numBands,numHops,1,numBkgClips,'single');

ind = 1;
for count = 1:numBkgFiles
    [wave,info] = read(ads);
    
    fs          = info.SampleRate;
    frameLength = frameDuration*fs;
    hopLength   = hopDuration*fs;
    
    for j = 1:numClipsPerFile(count)
        indStart =  randi(numel(wave)-fs);
        logVolume = logVolumeRange(1) + diff(logVolumeRange)*rand;
        volume = 10^logVolume;
        x = wave(indStart:indStart+fs-1)*volume;
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = auditorySpectrogram(x,fs, ...
            'WindowLength',frameLength, ...
            'OverlapLength',frameLength - hopLength, ...
            'NumBands',numBands, ...
            'Range',[0,8000], ...
            'WindowType','Hann', ...
            'WarpType','Mel', ...
            'SumExponent',2);
        
        if mod(ind,100)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end

disp("...done");

end