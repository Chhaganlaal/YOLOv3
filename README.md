# YOLOv3

A Tensorflow implementation of Joseph Redmond and Ali farhadi's You Only Look Once version 3. You only look once (YOLO) is a state-of-the-art, real-time object detection system. This repo consists only of the model and no training code. The model weights and the configuration file to build the model are taken from their own [site](https://pjreddie.com/darknet/yolo/).

#### [Paper: YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
#### [YOLO: Real Time Object Detection](https://pjreddie.com/darknet/yolo/)

![Detected](https://github.com/Chhaganlaal/YOLOv3/blob/master/result/det_person.jpg)

### Requirements
- Tensorflow
- Numpy
- OpenCV
- Pandas
- Matplotlib

### How to Run
1. Save images in the `image` folder.
2. Run `detect.py` script in console.
3. Result images are stored in `result` folder.

### Command Line Arguments
```
--images            Change input images' directory
--result            Change output images' directory
--nms_threshold     Non-Max Suppression Threshold (Default=0.5)
--confidence        Detection Confidence (Default=0.5)
--reso              Input image resolution for network (Default=320)
```
