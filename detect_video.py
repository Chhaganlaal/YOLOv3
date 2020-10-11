from __future__ import division
import gc
import time
import tensorflow as tf
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from DNModel import net
from img_process import preprocess_vid, inp_to_image
import pandas as pd
import random 
import pickle as pkl

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

def arg_parser():
    parser = argparse.ArgumentParser(description='YOLOv3 ')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing input images",
                        default = "images", type = str)
    parser.add_argument("--result", dest = 'result', help = 
                        " Directory to store results ",
                        default = "result", type = str)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.5)

    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Detection Confidence ", default = 0.5)
    parser.add_argument("--cfg", dest = 'configfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'resolution', help = 
                        "Input resolution of the network",
                        default = "320", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

def main():
    args = arg_parser()

    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    in_dim = int(args.resolution)

    num_classes = 80
    classes = load_classes('data/coco.names')

    model = net(args.configfile)
    model.load_weights(args.weightsfile)
    print("Network Loaded")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR opening camera")

    counter = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            processed_img, orig_img, im_dim = preprocess_vid(frame, in_dim)

            im_dim = tf.tile(tf.cast(im_dim, dtype=tf.float32), [2])

            if counter%5==0:
                prediction = model(processed_img)
            
                prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thresh)

                if prediction.shape[-1]!=8:
                    continue

                var_prediction = tf.Variable(prediction)

                im_dim_list = tf.tile(tf.expand_dims(im_dim, 0), [var_prediction.shape[0], 1])
                scaling_factor = tf.reshape(tf.math.reduce_min(in_dim/im_dim_list, axis=1), (-1,1))

                numpy_prediction = var_prediction.numpy()
                numpy_prediction[:, [1,3]] -= (in_dim-scaling_factor*im_dim_list[:,0].numpy().reshape(-1,1))/2
                numpy_prediction[:, [2,4]] -= (in_dim-scaling_factor*im_dim_list[:,1].numpy().reshape(-1,1))/2

                numpy_prediction[:, 1:5] /= scaling_factor

                numpy_im_dim_list = im_dim_list
                for i in range(numpy_prediction.shape[0]):
                    numpy_prediction[i, [1,3]] = np.clip(numpy_prediction[i, [1,3]], 0.0, numpy_im_dim_list[i,0])
                    numpy_prediction[i, [2,4]] = np.clip(numpy_prediction[i, [2,4]], 0.0, numpy_im_dim_list[i,1])
                output = tf.convert_to_tensor(numpy_prediction)

                colors = pkl.load(open("pallete", "rb"))

            def write_img(x, results):
                c1 = tuple(tf.cast(x[1:3], dtype=tf.int32))
                c2 = tuple(tf.cast(x[3:5], dtype=tf.int32))
                img = results

                cl = int(x[-1])
                label = f"{classes[cl]}"
                color = colors[0]
                cv2.rectangle(img, c1, c2, color, 1)
                font_scale = max(1, min(img.shape[0], img.shape[1])/(1000))
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, label, (c1[0], c1[1]+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, font_scale, [225,225,225], 1)
                return img

            counter += 1

            list(map(lambda x: write_img(x, orig_img), output))

            cv2.imshow("Frame", orig_img)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

        else:
            print("ERROR in reading")
            break

if __name__=='__main__':
    main()
