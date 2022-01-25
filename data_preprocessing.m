function [] = data_preprocessing()

ship_image = fullfile('E:\','\project','\train');
images = imageDatastore(ship_image, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

full_table = readtable('train_ship_segmentations_boxes.csv');

full_table(:,1) = []; %encoded pixel column
full_table(:,2) = []; % angle column

no_NaN_table = rmmissing(full_table); %remove nan rows

no_NaN_table.ImageId = fullfile(ship_image,no_NaN_table.ImageId); %add to path

no_NaN_table_2 = no_NaN_table;

no_NaN_table_2(300:end,:) = [];% 1000den sona kadar sil

%Removing duplicate images, -values and x values

bound_box_table = no_NaN_table_2;
bound_box_table(:,1) = []; % delete image ID column

bound_box_table = table2array(bound_box_table); %no round

bound_box_table2 = bound_box_table;
bound_box_table_round = round(bound_box_table2); %round 



bound_box_table2(:,5) = bound_box_table2(:,1);
bound_box_table2(:,6) = bound_box_table2(:,2);

bound_box_table_round(:,5) = bound_box_table_round(:,1); %change column
bound_box_table_round(:,6) = bound_box_table_round(:,2);

bound_box_table2(:,1) = []; % angle column
bound_box_table2(:,1) = []; % angle column %% 1 ve 2 sütunu, 3 ve 4 olarak

bound_box_table_round(:,1) = []; % 1.2. sutunu siliyor
bound_box_table_round(:,1) = [];

[W H] = size(bound_box_table2);

for i=1 : W
     cell_array_table(i,1) = table({bound_box_table2(i,:)});
end

for i=1 : W
     cell_array_table2(i,1) = table({bound_box_table_round(i,:)});
end

our_data = table(no_NaN_table_2.ImageId,cell_array_table2);
our_data.Var2 = table2cell(our_data.Var2);

bound_box_table_rounddddd = bound_box_table_round/2;
bound_box_table_rounddddd=round (bound_box_table_rounddddd);
for i=1 : W
     cell_array_table3(i,1) = table({bound_box_table_rounddddd(i,:)});
end
our_data4 = table(no_NaN_table_2.ImageId,cell_array_table3);
our_data4.Var2 = table2cell(our_data4.Var2);


