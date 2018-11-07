#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/19, grasses'

import tensorflow as tf, os

class Config():
    ROOT = os.path.abspath(os.path.dirname(__file__))

    # modeling
    label_size = 5
    batch_size = 256
    total_batch = 102400
    image_size = [64, 64, 3]
    optimizer = tf.train.AdamOptimizer
    scope_name = "dresden.model5"

    # leaning_rate
    decay_rate = 0.98
    decay_steps = 700
    learning_rate = 0.0018

    # generator
    test_list_size = 100
    min_after_dequeue = 40960
    dataset_name = "Dresden"
    input_path = os.path.join(ROOT, "dataset", dataset_name)
    output_path = os.path.join(ROOT, "dataset/output")

    train = {
        "summary_steps": 50,
        "saving_steps": 50
    }

    test = {

    }

    # Config for splicing plugin
    splicing = {
        "splice": 0.8,
        "stride": 32,
    }

    # Layer type. Don't change!!
    Conv, FC = "conv", "fc"

    net = {
        "01_Conv1": {
            "type": Conv,
            "patch_size": 8,
            "in_depth": 3,
            "out_depth": 16,
            "strides": [1, 1, 1, 1],
        },
        "02_Conv2": {
            "type": Conv,
            "patch_size": 8,
            "in_depth": 16,
            "out_depth": 32,
            "strides": [1, 2, 2, 1],
            "activation": tf.nn.relu,
        },
        "03_Conv3": {
            "type": Conv,
            "patch_size": 6,
            "in_depth": 32,
            "out_depth": 96,
            "strides": [1, 1, 1, 1],
        },
        "04_Conv4": {
            "type": Conv,
            "patch_size": 6,
            "in_depth": 96,
            "out_depth": 128,
            "strides": [1, 1, 1, 1],
            "activation": tf.nn.relu,
        },
        "05_Conv5": {
            "type": Conv,
            "patch_size": 3,
            "in_depth": 128,
            "out_depth": 256,
            "strides": [1, 2, 2, 1],
        },
        "06_Conv6": {
            "type": Conv,
            "patch_size": 3,
            "in_depth": 256,
            "out_depth": 512,
            "strides": [1, 1, 1, 1],
            "activation": tf.nn.relu,
        },
        "07_Conv7": {
            "type": Conv,
            "patch_size": 3,
            "in_depth": 512,
            "out_depth": 768,
            "strides": [1, 2, 2, 1],
            "activation": tf.nn.relu,
        },
        "08_Conv8": {
            "type": Conv,
            "patch_size": 3,
            "in_depth": 768,
            "out_depth": 512,
            "strides": [1, 1, 1, 1],
            "activation": tf.nn.relu,
        },
        "09_Conv9": {
            "type": Conv,
            "patch_size": 3,
            "in_depth": 512,
            "out_depth": 512,
            "strides": [1, 1, 1, 1],
            "activation": tf.nn.relu,
        },
        "10_Conv10": {
            "type": Conv,
            "patch_size": 1,
            "in_depth": 512,
            "out_depth": 256,
            "strides": [1, 2, 2, 1],
            "activation": tf.nn.relu,
        },
        "11_Conv11": {
            "type": Conv,
            "patch_size": 1,
            "in_depth": 256,
            "out_depth": 128,
            "strides": [1, 1, 1, 1],
            "dropout": 0.9,
            "activation": tf.nn.relu,
        },
        "12_Conv12": {
            "type": Conv,
            "patch_size": 1,
            "in_depth": 128,
            "out_depth": 32,
            "strides": [1, 2, 2, 1],
            "debug": True,
        },
        "13_FC1": {
            "type": FC,
            "in_depth": 128,
            "out_depth": 64,
            "dropout": 0.9,
        },
        "14_FC2": {
            "type": FC,
            "in_depth": 64,
            "out_depth": 32,
            "activation": tf.nn.relu,
        },
        "15_FC3": {
            "type": FC,
            "in_depth": 32,
            "out_depth": label_size,
            "debug": True,
        }
    }

    name_list = ["01_Conv1", "02_Conv2", "03_Conv3", "04_Conv4", "05_Conv5", "06_Conv6", "07_Conv7", "08_Conv8",
                 "09_Conv9", "10_Conv10", "11_Conv11", "12_Conv12", "13_FC1", "14_FC2", "15_FC3"]

    '''
    Note of decay dropout: 
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    decayed_dropout_rate = decayed_dropout + decay_impact * decayed_learning_rate
    '''