%% Define Parameters
% Feature extraction parameters
params.filter_size = [11 11];
params.color_space = 'ycbcr';
params.color_bins = 32;
params.grad_size = [16 16];
params.grad_bins = 9;
params.spatial_size = [32 32];
params.visualize = false;

% Other parameters
params.train_image_count = 22000;
params.test_image_count = 100;
params.test_image_display_count = 10;
params.window_size = [30 30];
params.num_steps = 70;
params.num_pca_features = 256;
params.pca_coeff = [];
params.confidence_thresh = 1.25;
params.heatmap_thresh = 0.3;
params.min_box_size = 20;
params.k_fold = 10;
params.poly_order = 3;

% Directory variables
params.root_dir = './Ship-Detection';
params.image_dir = 'train';

function [] = getBoundingBoxes(params)

    % Load input file
    input_file = fopen(fullfile(params.root_dir,'train_ship_segmentations_v2.csv'));
    
    % Read first header line (ignore it really)
    header = fgetl(input_file);
    
    % Create training file
    training_file = fopen(fullfile(params.root_dir, 'train_detections_dense1.csv'),'w');
    [train_ship_counter, train_no_ship_counter] = extractBoxes(input_file, ...
        training_file, params.root_dir, params.train_image_count);
    fclose(training_file);
    
    % Create testing file
    testing_file = fopen(fullfile(params.root_dir, 'test_detections_dense1.csv'),'w');
    [test_ship_counter, test_no_ship_counter] = extractBoxes(input_file, ...
        testing_file, params.root_dir, params.test_image_count);
    fclose(testing_file);
    
    fclose(input_file);
    
    disp("Positive/Negative sample ratio:");
    fprintf("Training: %i/%i\n",train_ship_counter,train_no_ship_counter);
    fprintf("Testing: %i/%i\n",test_ship_counter,test_no_ship_counter);
end