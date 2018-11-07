#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/19, grasses'

import tensorflow as tf, numpy as np, math, os, errno, shutil
curr_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.dirname(curr_path))

def to_obj(d):
    top = type("new", (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i, type(j)(to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def confidence(data):
    data = np.reshape(data, (-1, 3)) / 255.0
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    Q = 0
    for i in range(3):
        Q += 0.7 * 4 * (mean[i] - mean[i] * mean[i])
        Q += 0.3 * (1 - math.pow(math.e, math.log(0.01, math.e) * std[i]))
    Q = Q / 3.0
    return Q

'''
clean model && board
'''
def clean(base_path, scop_name):
    path = [os.path.join(base_path, "model", scop_name, "model"), os.path.join(base_path, "board", scop_name)]
    try:
        for item in path:
            for name in os.listdir(item):
                os.remove(os.path.join(item, name))
                print("=> clean: {:s}".format(os.path.join(item, name)))
    except Exception as e:
        print(e)

def backup(conf):
    try:
        shutil.copyfile(os.path.join(conf.ROOT, str(conf.__module__).split(".")[-1] + ".py"), os.path.join(conf.ROOT, "model", conf.scope_name, "model", "config"))
    except Exception as e:
        print("=> backup() e={:s}".format(str(e)))

def rebuild(model_path):
    # common path building in initing
    modules = ["pre-train", "splicing", "ground", "model"]
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for item in modules:
            if not os.path.exists(os.path.join(model_path, item)):
                os.makedirs(os.path.join(model_path, item))
    except OSError as e:
        print("=> [error] build_directory() e={:s}".format(e))
        if e.errno == errno.EEXIST and os.path.isdir(model_path):
            pass
        else:
            raise

def remove_dirty(base_path):
    black_list = [".DS_Store"]
    for name in black_list:
        path = "{:s}/{:s}".format(base_path, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception as e:
                print(e)
        if os.path.isdir(path):
            try:
                os.unlink(path)
            except Exception as e:
                print(e)

'''
:param value tfrecords int to init64
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
:param value tfrecords int to float32
'''
def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

'''
:param value tfrecords string to byte
'''
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

'''
normalize input X matrix value from 0~255 to -1~1
:param samples   numpy.array()
:return samples  numpy.array()
'''
def normalize(samples):
    return samples / 128.0 - 1.0

'''
reformat label Y matrix, for example: from Y=2 to Y=[0, 0, 1, 0, 0], where 1 is active value and label size is 5
:param labels    int
:return labels   numpy.array()
'''
def reformat(labels, label_size):
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * label_size
        if num == label_size:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return labels