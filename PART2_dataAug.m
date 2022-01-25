function [imageTrain12,imageValidation12,imageTest12,layers,options2,net3,accuracy2,plots2] = PART2_dataAug(imageTrain,imageValidation,imageTest)

% AYTEKÝN YILDIZHAN
% N18147923
% CMP719 Computer Vision Homework 1
% In this part, we train the net with data augmentation.

%deepNetworkDesigner

%----------------------------------------------------------------

% dogPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','dog');
% images = imageDatastore(dogPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% labelCount = countEachLabel(images)
% 
% 
% [imageTrain,imageValidation,imageTest] = splitEachLabel(images,0.6,0.2,'randomize');

%We do the data augmentation

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXReflection',true, ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30], ...
    'RandXScale',[0.9 1.1], ...
    'RandYScale',[0.9 1.1]);
    
imageTrain12 = augmentedImageDatastore([256,256],imageTrain, ...
    'DataAugmentation',imageAugmenter);

 imageValidation12 = augmentedImageDatastore([256,256],imageValidation, ...
     'DataAugmentation',imageAugmenter);

 imageTest12 = augmentedImageDatastore([256,256],imageTest, ...
     'DataAugmentation',imageAugmenter);

%Show the first eight augmented images
minibatch = preview(imageTrain12);
imshow(imtile(minibatch.input));
 

%building the layers
layers = [
    
    imageInputLayer([256 256 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
   % dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
  %  dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
   % dropoutLayer
    
    fullyConnectedLayer(20)
    softmaxLayer
    classificationLayer];


%tuning the hyperparameters
options2 = trainingOptions('adam', ...  
    'MiniBatchSize',64, ...    
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imageValidation12, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


%train the net
net3 = trainNetwork(imageTrain12,layers,options2);


%Calculate the validation accuracy
YTest = classify(net3,imageTest12);
TTest = imageTest.Labels;
 
accuracy2 = sum(YTest == TTest)/numel(TTest);

%plot the confusion matrix
plots2 = plotconfusion(TTest,YTest);

% Show the some of 4 example
% idx = randperm(numel(imageValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imageValidation,idx(i));
%     imshow(I)
%     label = YTest(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% end

end