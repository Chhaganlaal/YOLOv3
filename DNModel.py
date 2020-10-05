from __future__ import division

import gc
gc.collect()

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Concatenate, Input, LeakyReLU, UpSampling2D, Layer
from tensorflow.keras import Sequential, Model
# import tensorboard
import numpy as np
import cv2
from util import *


def construct_cfg(configFile):
    '''
    Build the network blocks using the configuration file.
    Pre-process it to form easy to manipulate using pytorch.
    '''

    config = open(configFile, 'r')
    file = config.read().split('\n')

    file = [line for line in file if len(line)>0 and line[0]!='#']
    file = [line.lstrip().rstrip() for line in file]

    networkBlocks = []
    networkBlock = {}

    for x in file:
        if x[0]=='[':
            if len(networkBlock)!=0:
                networkBlocks.append(networkBlock)
                networkBlock = {}
            networkBlock["type"] = x[1:-1].rstrip()
        else:
            entity, value = x.split('=')
            networkBlock[entity.rstrip()] = value.lstrip()
    networkBlocks.append(networkBlock)

    return networkBlocks

def get_anchors(anchors, masks):
    anchor_list = anchors.split(',')
    anchor_list = [int(a) for a in anchor_list]
    mask_list = masks.split(',')
    mask_list = [int(a) for a in mask_list]
    anchor_list = [(anchor_list[j], anchor_list[j+1]) for j in range(0, len(anchor_list), 2)]
    anchor_list = [anchor_list[j] for j in mask_list]

    return anchor_list

def fixed_padding(inputs, kernel_size, *args, mode="CONSTANT", **kwargs):
    '''
    Manually padding instead of using pad arguments
    '''
    pad_total = kernel_size - 1
    pad_beg = pad_total//2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0,0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)

    return padded_inputs

def build_network(networkBlocks):
    DNInfo = networkBlocks[0]
    modules = []
    anchor_lists = []
    outputs = []
    layer_outputs = {}

    inputs = Input(shape=[3,256,256], name="input")
    layer_inp = inputs

    for i, x in enumerate(networkBlocks[1:]):
        seq_module = Sequential()

        if x["type"]=="convolutional":

            seq_module._name = f"seq_conv_{i}"
            filters = int(x["filters"])
            pad = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            activation = x["activation"]

            if pad:
                padding = "same"
            else:
                padding = "valid"
            
            try:
                bn = int(x["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True
            
            conv = Conv2D(filters, kernel_size, stride, padding, data_format="channels_first", use_bias=bias, name=f"conv_{i}")
            seq_module.add(conv)

            if bn:
                bn = BatchNormalization(axis=1, momentum=0.9, epsilon=1e-05, name=f"batch_norm_{i}")
                seq_module.add(bn)
            
            if activation=="leaky":
                activn = LeakyReLU(alpha=0.1, name=f"leaky_{i}")
                seq_module.add(activn)

            layer_inp = seq_module(layer_inp)
            layer_outputs[i] = layer_inp

        elif x["type"]=="upsample":

            seq_module._name = f"seq_upsample_{i}"
            upsample = UpSampling2D(size=2, data_format="channels_first", interpolation="bilinear", name=f"upsample_{i}")
            seq_module.add(upsample)

            layer_inp = seq_module(layer_inp)
            layer_outputs[i] = layer_inp

        elif x["type"]=="route":
            
            x["layers"] = x["layers"].split(',')
            layers = [int(a) for a in x["layers"]]

            if layers[0]>0:
                layers[0] = layers[0] - i

            if len(layers)==1:
                layer_inp = layer_outputs[i+(layers[0])]

            else:
                if layers[1]>0:
                    layers[1] = layers[1] - i
                
                cat1 = layer_outputs[i+layers[0]]
                cat2 = layer_outputs[i+layers[1]]

                layer_inp = Concatenate(axis=1)([cat1, cat2])

            layer_outputs[i] = layer_inp

        elif x["type"]=="shortcut":
            
            from_ = int(x["from"])
            layer_inp = layer_outputs[i-1] + layer_outputs[i+from_]
            layer_outputs[i] = layer_inp

        elif x["type"]=="yolo":

            outputs.append(layer_outputs[i-1])
            layer_outputs[i] = layer_outputs[i-1]
            anchor_lists.append(get_anchors(x["anchors"], x["mask"]))

        modules.append(seq_module)

    model = Model(inputs=[inputs], outputs=outputs)

    # logdir="logs/"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # tensorboard_callback.set_model(model)

    print(model.summary())

    return (model, DNInfo, modules, anchor_lists)

class net(Layer):

    def __init__(self, cfgfile):
        super(net, self).__init__()

        self.netBlocks = construct_cfg(cfgfile)
        self.model, self.DNInfo, self.module_list, self.anchor_lists = build_network(self.netBlocks)
        self.header = tf.constant([0,0,0,0], dtype=tf.float32)
        self.seen = 0

    def call(self, x):
        detections = []
        modules = self.netBlocks[1:]
        layer_outputs = {}

        written_output = 0

        for i in range(len(modules)):
            module_type = modules[i]["type"]

            if module_type=="convolutional" or module_type=="upsample":
                x = self.module_list[i](x)
                layer_outputs[i] = x

            elif module_type=="route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if layers[0]>0:
                    layers[0] = layers[0] - i
                
                if len(layers)==1:
                    x = layer_outputs[i+(layers[0])]

                else:
                    if layers[1]>0:
                        layers[1] = layers[1] - i

                    map1 = layer_outputs[i+layers[0]]
                    map2 = layer_outputs[i+layers[1]]

                    x = tf.keras.layers.Concatenate(axis=1)([map1,map2])
                layer_outputs[i] = x

            elif module_type=="shortcut":
                from_ = int(modules[i]["from"])
                x = layer_outputs[i-1] + layer_outputs[i+from_]
                layer_outputs[i] = x

            elif module_type=="yolo":
                anchors = get_anchors(modules[i]["anchors"], modules[i]["mask"])
                inp_dim = int(self.DNInfo["height"])

                num_classes = int(modules[i]["classes"])

                # print("Shape before transform =>", x.shape)
                
                x = transformOutput(x, inp_dim, anchors, num_classes)

                # print("Shape after transform =>", x.shape)

                if type(x)==int:
                    continue

                if not written_output:
                    detections = x
                    written_output = 1
                
                else:
                    detections = Concatenate(axis=1)([detections, x])
                layer_outputs[i] = layer_outputs[i-1]

        try:
            return detections
        except:
            return 0

    # def call(self, x):
    #     print(x.shape)
    #     predictions_list = self.model(x)

    #     written_output = False
    #     for i, predictions in enumerate(predictions_list):
    #         anchors = self.anchor_lists[i]
    #         inp_dim = int(self.DNInfo["height"])
    #         num_classes = 80

    #         print(f"Shape before transform =>{predictions.shape}")

    #         predictions = transformOutput(predictions, inp_dim, anchors, num_classes)

    #         print(f"shape after transform =>{predictions.shape}")

    #         if type(predictions)==int:
    #             continue

    #         if not written_output:
    #             detections = predictions
    #             written_output = True
    #         else:
    #             detections = Concatenate(axis=1)([detections, predictions])

    #     try:
    #         return detections
    #     except:
    #         return 0
            

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = tf.convert_to_tensor(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        tracker = 0
        for i in range(len(self.module_list)):
            module_type = self.netBlocks[i+1]["type"]

            if module_type=="convolutional":
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.netBlocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv_part = model.get_layer(index=0)

                size = conv_part.kernel_size[0]
                channels = conv_part.input_shape[1]
                filters = conv_part.filters
                if batch_normalize:
                    bn_part = model.get_layer(index=1)

                    bias_count = filters

                    bn_bias = weights[tracker: tracker+bias_count]
                    tracker += bias_count

                    bn_weights = weights[tracker: tracker+bias_count]
                    tracker += bias_count

                    bn_running_mean = weights[tracker: tracker+bias_count]
                    tracker += bias_count

                    bn_running_var = weights[tracker: tracker+bias_count]
                    tracker += bias_count

                    bn_part.set_weights([bn_weights, bn_bias, bn_running_mean, bn_running_var])

                    weight_count = filters*size*size*channels

                    conv_weight = weights[tracker: tracker+weight_count].reshape(filters, channels, size, size)
                    conv_weight = np.transpose(conv_weight, [2,3,1,0])
                    tracker += weight_count

                    conv_part.set_weights([conv_weight])

                else:
                    bias_count = filters

                    conv_bias = weights[tracker: tracker+bias_count]
                    tracker += bias_count

                    weight_count = filters*size*size*channels

                    conv_weight = weights[tracker: tracker+weight_count].reshape(filters, channels, size, size)
                    conv_weight = np.transpose(conv_weight, [2,3,1,0])
                    tracker += weight_count

                    conv_part.set_weights([conv_weight, conv_bias])
        
        # print(tracker)
        assert tracker==len(weights), "failed to read all data"
                

'''
num_classes = 80
classes = load_classes("data/coco.names")

model = net("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network Loaded")

test_data = tf.random.normal([1,3,256,256], dtype=tf.float32)
test_output = model(test_data)

print(test_output.shape)
'''
            