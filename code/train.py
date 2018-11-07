#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/20, grasses'

from config import Config
from network import Network
from utils import Generator, tools
import tensorflow as tf

def main(conf=Config):
    tools.backup(conf)

    X = tf.placeholder(tf.float32, shape=[256, 64, 64, 3], name="X")
    Y = tf.placeholder(tf.float32, shape=[256, Config.label_size], name="Y")

    # Generator feeding data
    G = Generator(conf=conf)
    (G_X, G_Y, G_name, G_offset_x, G_offset_y, G_noise) = G.read()

    # Network building graph
    N = Network(conf=conf)
    N.train(X, Y, G_X, G_Y)

if __name__ == "__main__":
    main()