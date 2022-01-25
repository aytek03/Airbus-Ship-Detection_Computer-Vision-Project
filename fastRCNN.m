function [] = fastRCNN()

rng(0)
shuffledIdx = randperm(height(our_data));
idx = floor(0.6 * height(our_data));
trainingDataTbl = our_data(shuffledIdx(1:idx),:);
testDataTbl = our_data(shuffledIdx(idx+1:end),:);

imdsTrain = imageDatastore(trainingDataTbl{:,'Var1'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'Var2'));

imdsTest = imageDatastore(testDataTbl{:,'Var1'});
bldsTest = boxLabelDatastore(testDataTbl(:,'Var2'));

trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest,bldsTest);


featureExtractionNetwork = resnet50;


featureLayer = 'activation_40_relu';


numClasses = 1%;width(ourdata3)-1;

    
lgraph2 = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
%fast
options2 = trainingOptions('sgdm', ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 10, ...
    'CheckpointPath', tempdir);
%faster
options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 7, ...
      'VerboseFrequency', 200, ...
      'CheckpointPath', tempdir);

data = load('rcnnStopSigns.mat', 'stopSigns', 'fastRCNNLayers');
stopSigns = data.stopSigns;
fastRCNNLayers = data.fastRCNNLayers;

[frcnn,info] = trainFastRCNNObjectDetector(ds, fastRCNNLayers , options2);
[detector,info] = trainFasterRCNNObjectDetector(trainingData, lgraph2, options);

end