
from __future__ import division

import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  tf.math.maximum(b1_x1, b2_x1)
    inter_rect_y1 =  tf.math.maximum(b1_y1, b2_y1)
    inter_rect_x2 =  tf.math.minimum(b1_x2, b2_x2)
    inter_rect_y2 =  tf.math.minimum(b1_y2, b2_y2)
    
    inter_area = tf.math.maximum(inter_rect_x2-inter_rect_x1+1, tf.zeros(inter_rect_x2.shape))*tf.math.maximum(inter_rect_y2-inter_rect_y1+1, tf.zeros(inter_rect_x2.shape))

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


def transformOutput(prediction, inp_dim, anchors, num_classes):

    batch_size = prediction.shape[0]
    stride = inp_dim//prediction.shape[2]
    grid_size = prediction.shape[2]
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]


    prediction = tf.reshape(prediction, [batch_size, bbox_attrs*num_anchors, grid_size*grid_size])
    prediction = tf.transpose(prediction, [0,2,1])
    prediction = tf.reshape(prediction, [batch_size, grid_size*grid_size*num_anchors, bbox_attrs])


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    var_prediction = tf.Variable(prediction)
    var_prediction[:,:,0].assign(tf.math.sigmoid(prediction[:,:,0]))
    var_prediction[:,:,1].assign(tf.math.sigmoid(prediction[:,:,1]))
    var_prediction[:,:,4].assign(tf.math.sigmoid(prediction[:,:,4]))

    
    #Add the center offsets
    grid_len = np.arange(grid_size) + 1
    a,b = np.meshgrid(grid_len, grid_len)
    

    x_offset = tf.reshape(tf.convert_to_tensor(a, dtype=tf.float32), [-1,1])
    y_offset = tf.reshape(tf.convert_to_tensor(b, dtype=tf.float32), [-1,1])

    x_y_offset = tf.expand_dims(tf.reshape(tf.tile(tf.keras.layers.Concatenate(1)([x_offset, y_offset]), [1, num_anchors]), [-1,2]), 0)

    var_prediction[:,:,:2].assign(var_prediction[:,:,:2]+x_y_offset)
      
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    
        
    #Transform the anchors using log opearation given in research paper
    anchors = tf.expand_dims(tf.tile(anchors, [grid_size*grid_size, 1]), 0)
    var_prediction[:,:,2:4].assign(tf.math.exp(var_prediction[:,:,2:4])*anchors)

    #Softmax the class scores
    var_prediction[:,:,5: 5 + num_classes].assign(tf.math.sigmoid(var_prediction[:,:, 5 : 5 + num_classes]))

    var_prediction[:,:,:4].assign(var_prediction[:,:,:4]*stride)
   
    
    return tf.convert_to_tensor(var_prediction)

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = tf.convert_to_tensor(unique_np)
    
    tensor_res = tf.identity(tf.convert_to_tensor(unique_tensor))
    return tensor_res

def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = tf.expand_dims(tf.cast((prediction[:,:,4]>confidence), dtype=tf.float32), 2)
    
    var_prediction = tf.Variable(prediction*conf_mask)

    try:
        ind_nz = tf.identity(tf.transpose(tf.where(var_prediction[:,:,4]!=0)), [1,0])
    except:
        return 0
    
    
    box_a = tf.Variable(var_prediction)
    box_a[:,:,0].assign(var_prediction[:,:,0] - var_prediction[:,:,2]/2)
    box_a[:,:,1].assign(var_prediction[:,:,1] - var_prediction[:,:,3]/2)
    box_a[:,:,2].assign(var_prediction[:,:,0] + var_prediction[:,:,2]/2)
    box_a[:,:,3].assign(var_prediction[:,:,1] + var_prediction[:,:,3]/2)
    var_prediction[:,:,:4].assign(box_a[:,:,:4])

    
    batch_size = var_prediction.shape[0]
    
    output = tf.Variable(tf.random.normal([1, var_prediction.shape[2]+1]))
    write = False

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = var_prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf= tf.reduce_max(image_pred[:,5:5+num_classes], 1)
        max_conf_score = tf.math.argmax(image_pred[:,5:5+num_classes], 1)
        max_conf = tf.expand_dims(tf.cast(max_conf, dtype=tf.float32), 1)
        max_conf_score = tf.expand_dims(tf.cast(max_conf_score, dtype=tf.float32), 1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = tf.concat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind = tf.where(image_pred[:,4]!=0)
        
        image_pred_ = tf.reshape(image_pred.numpy()[tf.squeeze(non_zero_ind),:], [-1,7])
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cl in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*tf.expand_dims(tf.cast((image_pred_[:,-1] == cl), dtype=tf.float32), 1)            
            class_mask_ind = tf.squeeze(tf.where(cls_mask[:,-2]!=0))
            

            image_pred_class = tf.reshape(image_pred_.numpy()[class_mask_ind.numpy()], [-1,7])

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = tf.argsort(image_pred_class[:,4], direction="DESCENDING")
            np_image_pred_class = image_pred_class.numpy()
            
            var_image_pred_class = tf.Variable(np_image_pred_class[conf_sort_index.numpy()], dtype=tf.float32)
            idx = var_image_pred_class.shape[0]


            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        if var_image_pred_class.shape[0]>i+1:
                            ious = bbox_iou(tf.expand_dims(var_image_pred_class[i],0), var_image_pred_class[i+1:])
                        else:
                            break
                        
                    except ValueError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = tf.expand_dims(tf.cast((ious < nms_conf), dtype=tf.float32), 1)
                    var_image_pred_class[i+1:].assign(var_image_pred_class[i+1:]*iou_mask)
                    
                    #Remove the non-zero entries
                    non_zero_ind = tf.squeeze(tf.where(var_image_pred_class[:,4]!=0))
                    var_image_pred_class = tf.Variable(tf.reshape(var_image_pred_class.numpy()[non_zero_ind.numpy()], [-1,7]))
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = tf.fill([var_image_pred_class.shape[0], 1], ind*1.0)
            seq = batch_ind, tf.convert_to_tensor(var_image_pred_class)


            if not write:
                output = tf.keras.layers.Concatenate(1)(seq)
                write = True
            else:
                out = tf.keras.layers.Concatenate(1)(seq)
                output = tf.keras.layers.Concatenate(0)([output,out])

    return output



