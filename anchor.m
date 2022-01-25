function [] = anchor()

ship_image = fullfile('E:\','\project','\train');
images = imageDatastore(ship_image, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% full_table = readtable('train_ship_segmentations_boxes.csv'); %preprocess

full_table = readtable('train_detections_dense3.csv');

%full_table = readtable('train_ship_segmentations_v2.csv'); %encoded pixel

full_table2 = full_table; %keep original file

full_table(:,6) = []; % delete var6 column

%full_table = rmmissing(full_table); %remove nan rows

no_NaN_table = full_table;
no_NaN_table_2 = no_NaN_table;

no_NaN_table.Var1 = fullfile(ship_image,no_NaN_table.Var1); %add to path

no_NaN_table_2 = no_NaN_table;
no_NaN_table(5000:end,:) = [];% first 4999th data are train set

%Removing duplicate images, -values and x values

bound_box_table = no_NaN_table;
bound_box_table(:,1) = []; % delete image ID column

bound_box_table = table2array(bound_box_table); %change to array

bound_box_table2 = bound_box_table;%keep original file


%calculate width and height of bounding boxes
bound_box_table(:,3) = bound_box_table(:,3) - bound_box_table(:,1);
bound_box_table(:,4) = bound_box_table(:,4) - bound_box_table(:,2);
%-------------------------------------------------------

%change column 1st -> 2nd, 2st -> 1nd and 3rd -> 4th, 4st -> 3rd
bound_box_table(:,5) = bound_box_table(:,1);
bound_box_table(:,6) = bound_box_table(:,2);
bound_box_table(:,7) = bound_box_table(:,3);
bound_box_table(:,8) = bound_box_table(:,4);

bound_box_table(:,2) = bound_box_table(:,5);
bound_box_table(:,1) = bound_box_table(:,6);
bound_box_table(:,4) = bound_box_table(:,7);
bound_box_table(:,3) = bound_box_table(:,8);
%-------------------------------------------------------



%delete last four column
bound_box_table(:,8) = []; 
bound_box_table(:,7) = []; 
bound_box_table(:,6) = []; 
bound_box_table(:,5) = []; 
%-------------------------------------------------------

%array to cell
[W H] = size(bound_box_table2);


for i=1 : W
     cell_array_table(i,1) = table({bound_box_table(i,:)});
end

%-------------------------------------------------------
%add to path of train file
our_data_test2 = table(no_NaN_table.Var1,cell_array_table);
our_data_test2.Var2 = table2cell(our_data_test2.Var2);

%image and box label
imds = imageDatastore(our_data_test2.Var1);
blds = boxLabelDatastore(our_data_test2(:,2:end));
ds = combine(imds, blds);
%-------------------------------------------------------
%DATA AUGMENTATION PART
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXReflection',true, ...
%     'RandXTranslation',[-30 30], ...
%     'RandYTranslation',[-30 30], ...
%     'RandXScale',[0.9 1.1], ...
%     'RandYScale',[0.9 1.1]);
%  
% imds3 = augmentedImageDatastore([256,256],imds, ...
%     'DataAugmentation',imageAugmenter);
% minibatch = preview(imds3);
% imshow(imtile(minibatch.input));



%reading data...
data = read(ds);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
%annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%-------------------------------------------------------


%our input size
inputSize = [256,256,3];
preprocessedTrainingData = transform(ds, @(data)preprocessData(data,inputSize));

%anchor boxes
numAnchors = 3;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

numClasses = 1;
network = resnet50();
%network = googlenet;
%analyzeNetwork(network);
featureLayer = 'activation_49_relu';
%featureLayer = 'inception_4d-output';
%inceptionv3	'mixed7'
% vgg19 relu5_4
% mobilenetv2	'block_13_expand_relu'
%vgg16	'relu5_3'
%inceptionresnetv2	'block17_20_ac'
%-------------------------------------------------------

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,network, featureLayer);
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',8,...
          'MaxEpochs',7,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);
      
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options)      
%-------------------------------------------------------

%results = detect(detector, imds);
% C = table2cell(results);
% C(~cellfun('isempty',C));

figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')

% allBoxes = vertcat(our_data_test2.Var2{:});
% aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
% area = prod(allBoxes(:,3:4),2);
% 
% figure
% scatter(area,aspectRatio)
% xlabel("Box Area")
% ylabel("Aspect Ratio (width/height)");
% title("Box Area vs. Aspect Ratio")
% 
% 
% maxNumAnchors = 20;
% meanIoU = zeros([maxNumAnchors,1]);
% anchorBoxes = cell(maxNumAnchors, 1);
% for k = 1:maxNumAnchors
%     % Estimate anchors and mean IoU.
%     [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(blds2,k);    
% end
% 
% figure
% plot(1:maxNumAnchors,meanIoU,'-o')
% ylabel("Mean IoU")
% xlabel("Number of Anchors")
% title("Number of Anchors vs. Mean IoU")

%img = imread('E:\project\train\000155de5.jpg');
%img = imread('E:\project\train\00269a792.jpg');
img = imread('E:\project\test\0b8cde107.jpg');
%img1 = imread('C:\Users\Aytekin\Desktop\Ship-Detection-Project-master\panama_canal_test\test_image13.png');
%img = imread('E:\project\vehicleImages\image_00001.jpg');
[bboxes,scores] = detect(detector2,img);

% detectionResults = detect(detector,imdsTest)
if(~isempty(bboxes))
    img = insertObjectAnnotation(img,'rectangle',bboxes,scores);
end
figure
imshow(img)

end