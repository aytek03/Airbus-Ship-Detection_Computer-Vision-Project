function [] = yolo()

imds = imageDatastore(our_data.Var1);
blds = boxLabelDatastore(our_data(:,2:end));
ds = combine(imds, blds);

% imds = imageDatastore(gTruth2.imageFilename);
% blds = boxLabelDatastore(gTruth2(:,2:end));

inputSize = [256,256,3];
preprocessedTrainingData = transform(ds, @(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);
 I = data{1};
 bbox = data{2};
 
%  data = readall(preprocessedTrainingData);
%  I = data{1};
%  bbox = data{2};

numAnchors = 1;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

%inputSize = [256,256,3];
numClasses = 1;
network = resnet50();
featureLayer = 'activation_49_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,network, featureLayer)

allBoxes = vertcat(gTruth.Var2{:});
aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);

figure
scatter(area,aspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio")

trainingData = boxLabelDatastore(our_data6(:,2:end));

rng(0);
shuffledIdx = randperm(height(gTruth));
gTruth = gTruth(shuffledIdx,:);

imds = imageDatastore(our_data.Var1);
blds = boxLabelDatastore(our_data(:,2:end));
ds = combine(imds, blds);

data = read(ds);


I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


imds2 = imageDatastore(our_data6.Var1);
blds2 = boxLabelDatastore(our_data6(:,2:end));
ds2 = combine(imds2, blds2);

net = load('yolov2VehicleDetector.mat');
lgraph = net.lgraph

options = trainingOptions('adam',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',4,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);
     % analyzeNetwork(lgraph);

     
inputSize = [512 512 3];
preprocessedTrainingData = transform(ds, @(data)preprocessData(data,inputSize));
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options)

figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')

imdsTest = imageDatastore(our_data_test.Var1);
bldsTest = boxLabelDatastore(our_data_test(:,2:end));
ds1 = combine(imdsTest, bldsTest);

prep_testdata = transform(ds1,@(data)preprocessData(data,inputSize));

results = detect(detector, imdsTest);

[ap, recall, precision] = evaluateDetectionPrecision(results, bldsTest);
figure;
plot(recall, precision);
grid on
title(sprintf('Average precision = %.1f', ap))
figure

[am, fppi, missRate] = evaluateDetectionMissRate(results, bldsTest);
figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))




img = imread('E:\project\train\000155de5.jpg');

 img = imread('E:\project\train\001e418bc.jpg');
% 
% img = imread('C:\Users\Aytekin\Desktop\Ship-Detection-Project-master\panama_canal_test\test_image13.png');
% img = imread('E:\project\vehicleImages\image_00001.jpg');
% img = imread('E:\project\train\0002756f7.jpg');

%img = imread('E:\project\train\0006c52e8.jpg');
%img = imread('E:\project\train\002deeb16.jpg');

%img = imread('E:\project\train\0060f6de0.jpg');

[bboxes,scores] = detect(detector,img);

% bboxes = [169,482,181,36];
% scores = 0.59427190;

%if(~isempty(bboxes))
    img = insertObjectAnnotation(img,'rectangle',bboxes,scores);
%end
figure
imshow(img)

data = read(ds);
I = data{1};
bbox = data{2};
%bbox = [298,334,307,45]
annotatedImage = insertShape(I,'Rectangle',bbox);
%annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
end

load('our_data_test2.mat')
our_data_test2(1:81,:) = []; 
our_data_test2(28:end,:) = [];

imdsTest = imageDatastore(our_data_test2.Var1);
bldsTest = boxLabelDatastore(our_data_test2(:,2:end));
ds1 = combine(imdsTest, bldsTest);
results = detect(detector, imdsTest)
results2 = detect(detector, ds1)

targetSize = [600,600,3];
imdsReSz = transform(imdsTest,@(data) imsharpen(data));
results = detect(detector, imdsReSz)
results2 = detect(detector, ds1)

