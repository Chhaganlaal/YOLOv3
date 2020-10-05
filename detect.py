from __future__ import division
import gc
import time
import tensorflow as tf
# from absl import app, flags, logging
# from absl.flags import FLAGS
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from DNModel import net
from img_process import preprocess_img, inp_to_image
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

if __name__=='__main__':
    args = arg_parser()

    scales = args.scales

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0

    CUDA = False

    num_classes = 80
    classes = load_classes('data/coco.names')

    model = net(args.configfile)
    model.load_weights(args.weightsfile)
    print("Network Loaded")

    # model.DNInfo["height"] = args.resolution
    in_dim = int(args.resolution)

    read_dir = time.time()
    try:
        print(os.path)
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1]=='.png' or os.path.splitext(img)[1]=='.jpeg' or os.path.splitext(img)[1]=='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print(f"No file with the name{images}")
        exit()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    batches = list(map(preprocess_img, imlist, [in_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_img = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]

    im_dim_list = tf.tile(tf.cast(im_dim_list, dtype=tf.float32), [1,2])
    

    leftover = 0

    if len(im_dim_list)%batch_size:
        leftover = 1

    i = 0

    write = False

    objs = {}

    for batch in im_batches:
        print('batch size =>', batch.shape)

        prediction = model(batch)

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thresh)


        if prediction.shape[-1] != 8:
            i += 1
            continue

        var_prediction = tf.Variable(prediction)
        var_prediction[:,0].assign(var_prediction[:,0]+i*batch_size)

        if not write:
            output = var_prediction
            write = True
        else:
            output = tf.keras.layers.Concatenate(0)([output, var_prediction])
        print(output.shape)

        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in tf.convert_to_tensor(output) if int(x[0])==im_id]
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    print(im_dim_list.shape)        
    im_dim_list = tf.gather(im_dim_list, tf.cast(output[:,0], dtype=tf.int64), axis=0)
    print(im_dim_list.shape)
    scaling_factor = tf.reshape(tf.math.reduce_min(in_dim/im_dim_list, axis=1), (-1,1))
    print(scaling_factor.shape)

    # Workaround
    numpy_output = output.numpy()
    numpy_output[:, [1,3]] -= (in_dim-scaling_factor*im_dim_list[:,0].numpy().reshape(-1,1))/2
    numpy_output[:, [2,4]] -= (in_dim-scaling_factor*im_dim_list[:,1].numpy().reshape(-1,1))/2

    numpy_output[:, 1:5] /= scaling_factor

    numpy_im_dim_list = im_dim_list
    for i in range(numpy_output.shape[0]):
        numpy_output[i, [1,3]] = np.clip(numpy_output[i, [1,3]], 0.0, numpy_im_dim_list[i,0])
        numpy_output[i, [2,4]] = np.clip(numpy_output[i, [2,4]], 0.0, numpy_im_dim_list[i,1])
    output = tf.convert_to_tensor(numpy_output)

    colors = pkl.load(open("pallete", "rb"))

    def write_img(x, results):
        c1 = tuple(tf.cast(x[1:3], dtype=tf.int32))
        c2 = tuple(tf.cast(x[3:5], dtype=tf.int32))
        img = results[int(x[0])]

        cl = int(x[-1])
        label = f"{classes[cl]}"
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        font_scale = max(1, min(img.shape[0], img.shape[1])/(1000))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1]+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, font_scale, [225,225,225], 1)
        return img

    list(map(lambda x: write_img(x, orig_img), output))

    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.result, x.split('\\')[-1]))

    list(map(cv2.imwrite, det_names, orig_img))

    gc.collect()
