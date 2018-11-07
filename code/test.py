#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/20, grasses'

from config import Config
from network import Network
from utils.splicing import Splicing
import os, tensorflow as tf, numpy as np, pandas as pd, utils.tools as utils
curr_path = os.path.dirname(os.path.realpath(__file__))

class Test(object):
    def __init__(self, net=Network(conf=Config)):
        self.net = net
        self._conf = net._conf
        self.save_path = net.save_path
        self.model_path = os.path.join(curr_path, "model", self._conf.scope_name)

        # network params
        self.X = tf.placeholder(tf.float32, shape=[self._conf.batch_size, 64, 64, 3], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[self._conf.batch_size, self._conf.label_size], name="Y")

    def run(self, iterator, is_true_test=False):
        if not self.net.has_model():
            print("=> Please train your model first!!")
            exit(1)

        # build graph
        self.net.model(self.X).optimizer(self.Y)
        with tf.Session() as session:
            # restore session
            print("=> loading session from: {:s}".format(self.net.save_path))
            self.net.saver.restore(session, self.net.save_path)

            # file iterator
            pre_output = []

            for (index, flag, name, X, Y, Q, tamper_rate, shape, offset_x, offset_y, width, height) in iterator():
                (_X, _Y) = (utils.normalize(X), utils.reformat(Y, self._conf.label_size))
                (params, softmax, loss, predict, accuracy) = session.run(
                    [self.net.params_value, self.net.output, self.net.loss, self.net.predict, self.net.accuracy],
                    feed_dict={
                        self.X: _X,
                        self.Y: _Y
                    }
                )
                # tmp = [softmax(*label_size), predict, label, confidence]
                tmp = np.zeros([256, 8], dtype=np.float32)
                tmp[:, 0: 5] = softmax
                tmp[:, 5] = predict
                tmp[:, 6] = Y
                tmp[:, 7] = Q
                tmp = np.reshape(tmp, (-1, 8))
                pre_output.append(tmp)

                # save result into csv file
                if index == flag:
                    pre_output = np.array(pre_output, dtype=np.float32)
                    name = "{:s}_{:d}_{:d}".format(name, width, height)
                    df = pd.DataFrame(np.reshape(pre_output, (-1, 8)))
                    print(df.shape)

                    if not is_true_test:
                        csv_path = os.path.join(self.model_path, "pre-train", "{:s}.csv".format(name))
                    else:
                        csv_path = os.path.join(self.model_path, "pre-train-true", "{:s}.csv".format(name))
                    print("=> save pretrain at csv_path={:s}".format(csv_path))
                    df.to_csv(csv_path)
                    pre_output = []
                    
                for i in range(len(predict)):
                    if predict[i] != Y[i]:
                        print("=> index={:d} p={:d} Y={:d} tamper_rate={:.5f} output={:s}".format(i, predict[i], Y[i], tamper_rate[i], softmax[i]))
                print("=> ({:d}, {:d}), loss={:.5f}, accuracy:{:.3f}% \n".format(index, flag, loss, accuracy))
                if len(params) > 0: print("\n\n")
                
            print("=> pretrain csv format={f1,f2,f3...,predict,label,texture_quality}, f1,f2 is CNN confidence for each model\n")

def main(conf=Config):
    S = Splicing(conf=conf)
    T = Test(net=Network(conf=conf))
    T.run(S.iterator)

if __name__ == "__main__":
    main()