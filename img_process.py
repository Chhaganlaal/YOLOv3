from __future__ import division

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image


def custom_resize(img, inp_dim):
    '''
    Resize without changing aspect ratio
    '''

    img_h, img_w = img.shape[0], img.shape[1]
    w, h = inp_dim
    new_h = int(img_h*min(w/img_w, h/img_h))
    new_w = int(img_w*min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Resized image on a gray background
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    # cv2.imshow('canvas', np.array(canvas, dtype=np.uint8))
    # cv2.waitKey(0)
    return canvas

def preprocess_img(img, inp_dim):
    '''
    Preprocess the image for neural network.
    Returns a Tensor.
    '''

    orig_img = cv2.imread(img)
    dim = orig_img.shape[1], orig_img.shape[0]
    img = (custom_resize(orig_img, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = tf.keras.backend.expand_dims(tf.divide(tf.convert_to_tensor(img_, dtype=tf.float32), tf.constant(255.0)), axis=0)

    return img_, orig_img, dim

def preprocess_vid(frame, inp_dim):

    orig_img = frame
    dim = orig_img.shape[1], orig_img.shape[0]
    img = (custom_resize(orig_img, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = tf.keras.backend.expand_dims(tf.divide(tf.convert_to_tensor(img_, dtype=tf.float32), tf.constant(255.0)), axis=0)

    return img_, orig_img, dim

def inp_to_image(inp):
    inp = tf.squeeze(inp)
    inp = tf.multiply(inp, 255)

    inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]

    cv2.imshow('inp', inp)
    cv2.waitKey(0)
    return inp

