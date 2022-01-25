function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.

%  data = read(data);
%  I = data{1};
%  bbox = data{2};

scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);

end
% data = read(trainingData);
% I = data{1};
% bbox = data{2};
% annotatedImage = insertShape(I,'Rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)