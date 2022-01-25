# Airbus-Ship-Detection_Computer-Vision-Project

Deep Learning (DL) is truly helpful techniques for
image detection, segmentation and classification etc. In this
project, we analyze the Airbus Ship dataset and detect the ships
in the satellite images. Firstly, we do some preprocessing the
data. There is no bounding box information in our data set.
Only mask information is included. We convert masks into the
bounding boxes. Then, we estimate anchor boxes and we train the
data with YOLOv2 object detector by using these anchor boxes.
We use six different networks which are ResNet-50, GoogleNet,
MobileNet-V2, VGG16, VGG19, and InceptionResNet-V2. Only
InceptionResNet-V2 could not detect any ships on the test set.
Secondly, we use three pre-trained network systems trained on
PASCAL VOC2007 dataset. These are Fast-RCNN, Faster-RCNN
and Single Shot MultiBox Detector (SSM). We download and
install MatConvNet’s library and we just test the single image.
Finally, we show the results and discussion about future work.
