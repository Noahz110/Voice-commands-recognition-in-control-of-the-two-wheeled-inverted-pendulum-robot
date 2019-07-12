%% Thay doi duong dan den thu muc chua mau huan luyen
datafolder = fullfile('E:\Documents\xxxxxx\New folder\','speech_commands');

addpath(fullfile(matlabroot,'toolbox','audio','audiodemos'))
ads = audioexample.Datastore(datafolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames', ...
    'ReadMethod','File')
ads0 = copy(ads);

%% Chon ten cac thu muc can de nhan dang
commands = ["Tien","Lui","Trai","Phai","Dung"];

isCommand = ismember(ads.Labels,categorical(commands));
isUnknown = ~ismember(ads.Labels,categorical([commands,"_background_noise_"]));

probIncludeUnknown = 0.2;
mask = rand(numel(ads.Labels),1) < probIncludeUnknown;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

ads = getSubsetDatastore(ads,isCommand|isUnknown);
countEachLabel(ads)

%% Chia tap mau ra thanh tap huan luyen va tap xac nhan mang
[adsTrain,adsValidation] = splitEachLabel(ads,0.8,'randomized');

%% Tinh toan quang pho giong noi
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;

addpath(fullfile(matlabroot,'examples','audio','main'))
epsil = 1e-6;

XTrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
XTrain = log10(XTrain + epsil);

XValidation = speechSpectrograms(adsValidation,segmentDuration,frameDuration,hopDuration,numBands);
XValidation = log10(XValidation + epsil);

YTrain = adsTrain.Labels;
YValidation = adsValidation.Labels;

specMin = min(XTrain(:));
specMax = max(XTrain(:));
%% Bo sung them background
adsBkg = getSubsetDatastore(ads0, ads0.Labels=="_background_noise_");
numBkgClips = 1500;
volumeRange = [1e-4,1];

XBkg = backgroundSpectrograms(adsBkg,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands);
XBkg = log10(XBkg + epsil);

numTrainBkg = floor(0.8*numBkgClips);
numValidationBkg = floor(0.2*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = XBkg(:,:,:,1:numTrainBkg);
XBkg(:,:,:,1:numTrainBkg) = [];
YTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValidationBkg) = XBkg(:,:,:,1:numValidationBkg);
clear XBkg
YValidation(end+1:end+numValidationBkg) = "background";

YTrain = removecats(YTrain);
YValidation = removecats(YValidation);
%% Bien doi mau huan luyen cho tang do da dang
sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter(...
    'RandXTranslation',[-10 10],...
    'RandXScale',[0.8 1.2],...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain,...
    'DataAugmentation',augmenter);

%% Cau truc mang CNN
classNames = categories(YTrain);
classWeights = 1./countcats(YTrain);
classWeights = classWeights/mean(classWeights);
numClasses = numel(classNames);

timePoolSize = ceil(imageSize(2)/8);
dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
   
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([1 timePoolSize])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedCrossEntropyLayer(classNames,classWeights)];

%% Huan luyen mang
miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',Inf, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'ExecutionEnvironment','gpu');

trainedNet = trainNetwork(augimdsTrain,layers,options);
%% Kiem tra mang da huan luyen

YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

figure
plotconfusion(YValidation,YValPred,'Validation Data')