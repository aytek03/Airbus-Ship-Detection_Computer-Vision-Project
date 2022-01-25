function [] = anchor_test(no_NaN_table,inputSize)

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

no_NaN_table_2 = no_NaN_table; %keep original file
%no_NaN_table = no_NaN_table_2;

no_NaN_table(1:4999,:) = [];
no_NaN_table_2(1:4999,:) = [];
% no_NaN_table = removevars(no_NaN_table, 'Var2');
% no_NaN_table = removevars(no_NaN_table, 'Var3');
% no_NaN_table = removevars(no_NaN_table, 'Var4');
% no_NaN_table = removevars(no_NaN_table, 'Var5');
no_NaN_table(2143:end,:) = [];

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
our_data_test3 = table(no_NaN_table.Var1,cell_array_table);
our_data_test3.Var2 = table2cell(our_data_test3.Var2);


inputSize = [256,256,3];

our_data_test4 = our_data_test3;
%our_data_test3 = our_data_test4;
our_data_test3(100:end,:) = [];


imds2 = imageDatastore(our_data_test3.Var1);
blds2 = boxLabelDatastore(our_data_test3(:,2:end));
ds2 = combine(imds2, blds2);
preprocessedTestData = transform(ds2,@(data)preprocessData(data,inputSize));

results = detect(detector2, preprocessedTestData); %googlenet çalýþtý


[ap,recall,precision] = evaluateDetectionPrecision(results, preprocessedTestData);


figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

[am, fppi, missRate] = evaluateDetectionMissRate(results, blds2);
figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))
end