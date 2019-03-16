#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/19, grasses'

from code.config import Config
import code.utils.tools as utils, time, os, tensorflow as tf, numpy as np
curr_path = os.path.dirname(os.path.realpath(__file__))

class Network(object):
    def __init__(self, conf=Config):
        self._conf = conf

        # tensorboard
        self.merged = None
        self.summaries = []
        self.merged_summary = []
        self.writer = None
        self.writer_path = "{:s}/board/{:s}".format(curr_path, self._conf.scope_name)

        # model saver
        self.saver = None
        self.save_path = "{:s}/model/{:s}/model/model.ckpt".format(curr_path, self._conf.scope_name)

        # params
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.params_key = []
        self.params_value = []

        # network common params
        self.X = None
        self.Y = None
        self.logits = None

        # init function
        self.model_path = "{:s}/model/{:s}/".format(curr_path, self._conf.scope_name)
        utils.rebuild(self.model_path)

    def apply_regularization(self, _lambda=5e-4):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        return _lambda * regularization

    def xavier_init(self, in_size, out_size, constant=1.0):
        # for detail see paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        low = -constant * np.sqrt(6.0 / (in_size + out_size))
        high = constant * np.sqrt(6.0 / (in_size + out_size))
        return tf.random_uniform((in_size, out_size), minval=low, maxval=high, dtype=tf.float32)

    def add_cnn(self, name, kernel, value=0.1, stddev=0.1):
        '''
        :param name:    layer name
        :param kernel:  kernel size [patch_size, in_size, out_size]
        :param stddev:  stddev of w
        :return: (w, b)
        '''
        w = tf.Variable(tf.truncated_normal(kernel, stddev=stddev), name="{:s}_weights".format(name))
        b = tf.Variable(tf.constant(value=value, shape=[kernel[3]]), name="{:s}_biases".format(name))
        self.conv_weights.append(w)
        self.conv_biases.append(b)
        return (w, b)

    def add_fc(self, name, in_depth, out_depth, value=0.1):
        '''
        :param name:        layer name
        :param in_depth:    fully connected in size
        :param out_depth:   fully connected out size
        :param value:       init value
        :return:            (w, b)
        '''
        w = tf.Variable(self.xavier_init(in_depth, out_depth), name="{:s}_weights".format(name))
        b = tf.Variable(tf.constant(value=value, shape=[out_depth], name="{:s}_biases".format(name)))
        self.fc_weights.append(w)
        self.fc_biases.append(b)
        return (w, b)

    def has_model(self):
        return os.path.exists(self.save_path + ".meta")

    def model(self, X, is_train=True):
        seed = int(time.time())
        print("=> X={:s}".format(X.get_shape().as_list()))

        with tf.name_scope("leaning_rate"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self._conf.learning_rate,
                global_step=self.global_step,
                decay_steps=self._conf.decay_steps,
                decay_rate=self._conf.decay_rate,
                staircase=True
            )
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            # decayed_dropout_rate = base + decay_impact * decayed_learning_rate

        self.X = X

        # for (name, _) in sorted(self._conf.net.items(), key=lambda x: x[0: 2]):
        # item = utils.to_obj(_)

        for name in self._conf.name_list:
            _ = self._conf.net[name]
            item = utils.to_obj(_)

            with tf.name_scope(name):
                if item.type == Config.Conv:
                    (w, b) = self.add_cnn(name, [item.patch_size, item.patch_size, item.in_depth, item.out_depth])
                    X = tf.nn.conv2d(X, filter=w, strides=item.strides, padding="SAME") + b
                elif item.type == Config.FC:
                    shape = X.get_shape().as_list()
                    if len(shape) > 2:
                        X = tf.reshape(X, [-1, shape[1] * shape[2] * shape[3]])
                    (w, b) = self.add_fc(name, item.in_depth, item.out_depth)
                    X = tf.matmul(X, w) + b

                # activation layer
                if _.has_key("activation") and _["activation"]:
                    X = _["activation"](X)

                # pooling layer
                if _.has_key("pooling") and _["pooling"]:
                    X = _["pooling"](
                        X,
                        ksize=item.pooling_ksize,
                        strides=item.pooling_strides,
                        padding=item.pooling_padding
                    )

                # dropout layer
                if _.has_key("dropout") and _["dropout"]:
                    X = tf.nn.dropout(X, keep_prob=item.dropout, seed=seed) if is_train else X / item.dropout

                # decay dropout
                if _.has_key("decay_dropout"):
                    dropout = item.decay_dropout + ((1 - item.decay_dropout) * (self.learning_rate/ self._conf.learning_rate))
                    X = tf.nn.dropout(X, keep_prob=dropout, seed=seed) if is_train else X / dropout
                    self.summaries.append(tf.summary.scalar("decay_dropout", dropout))

                # save in debug params
                if _.has_key("debug") and _["debug"]:
                    self.params_key.append(name)
                    self.params_value.append(X)

            print("=> name={:s} X={:s}".format(name, X.get_shape().as_list()))
        self.logits = X
        return self

    def optimizer(self, Y):
        self.Y = Y
        with tf.name_scope("loss"):
            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            self.loss += self.apply_regularization(_lambda=5e-4)

        with tf.name_scope("accuracy"):
            # prediction
            self.output = tf.nn.softmax(logits=self.logits, name="output")
            self.predict = tf.argmax(self.output, axis=1)

            # accuracy
            accuracy_op = tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.Y, axis=1))
            self.accuracy = 100 * tf.reduce_mean(tf.cast(accuracy_op, tf.float32))

        with tf.name_scope("optimizer"):
            if self._conf.optimizer == tf.train.MomentumOptimizer:
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.5).minimize(self.loss, global_step=self.global_step)
            elif self._conf.optimizer == tf.train.AdamOptimizer:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss, global_step=self.global_step)

        # add summary
        self.summaries.append(tf.summary.scalar("accuracy", self.accuracy))
        self.summaries.append(tf.summary.scalar("loss", self.loss))
        # merged summary
        self.merged_summary = tf.summary.merge(self.summaries)
        # save graph
        self.saver = tf.train.Saver(tf.global_variables())
        return self

    def train(self, X, Y, G_X, G_Y):
        '''
        :param X:   placeholder for X
        :param Y:   placeholder for Y
        :param G_X:     tfrecords generator X
        :param G_Y:     tfrecords generator Y
        :param G_name:  tfrecords generator name
        :return:        null
        '''
        self.model(X).optimizer(Y)
        with tf.Session() as session:
            self.writer = tf.summary.FileWriter(logdir=self.writer_path, graph=session.graph)
            if not self.has_model():
                session.run(tf.global_variables_initializer())
            else:
                print("=> loading session from: {:s}".format(self.save_path))
                self.saver.restore(session, self.save_path)

            tf.train.start_queue_runners(sess=session, coord=tf.train.Coordinator())
            global_step = session.graph.get_tensor_by_name("leaning_rate/global_step:0")
            for index in range(global_step.eval(), self._conf.total_batch):
                (samples, labels) = session.run([G_X, G_Y])
                # _X = np.reshape(G.normalize(samples), [self.batch_size, self.image_size * self.image_size * 3])
                (_X, _Y) = (utils.normalize(samples), utils.reformat(labels, self._conf.label_size))

                (_, params, predict, loss, accuracy, output, summary) = session.run(
                    [self.optimizer, self.params_value, self.predict, self.loss, self.accuracy, self.output, self.merged_summary],
                    feed_dict={
                        self.X: _X,
                        self.Y: _Y
                    }
                )

                if index % self._conf.train["summary_steps"] == 0:
                    self.writer.add_summary(summary, index)
                    self.writer.flush()
                if index % self._conf.train["saving_steps"] == 0:
                    for i in range(len(self.params_key)):
                        print("=> {:s} -> {:s}".format(self.params_key[i], str(params[i][0])))
                    print("=> step={:d}, loss={:.5f}, accuracy:{:.3f}%".format(index, loss, accuracy))
                    if len(params) > 0:
                        print("\n\n\n")
                    self.saver.save(session, self.save_path)

    def iterator_train(self, X, Y, iterator):
        self.model(X).optimizer(Y)
        with tf.Session() as session:
            self.writer = tf.summary.FileWriter(logdir=self.writer_path, graph=session.graph)
            if not self.has_model():
                session.run(tf.global_variables_initializer())
            else:
                print("=> loading session from: {:s}".format(self.save_path))
                self.saver.restore(session, self.save_path)

            global_step = session.graph.get_tensor_by_name("leaning_rate/global_step:0")
            for index in range(global_step.eval(), self._conf.total_batch):
                for (count, step_count, samples, labels) in iterator():
                    (_X, _Y) = (utils.normalize(samples), utils.reformat(labels, self._conf.label_size))

                    (_, params, predict, loss, accuracy, output, summary) = session.run(
                        [self.optimizer, self.params_value, self.predict, self.loss, self.accuracy, self.output, self.merged_summary],
                        feed_dict={
                            self.X: _X,
                            self.Y: _Y
                        }
                    )
                    if step_count % self._conf.train["summary_steps"] == 0:
                        self.writer.add_summary(summary, index)
                        self.writer.flush()
                    if step_count % self._conf.train["saving_steps"] == 0:
                        for i in range(len(self.params_key)):
                            print("=> {:s} -> {:s}".format(self.params_key[i], str(params[i][0])))
                        print("=> step={:d}, loss={:.5f}, accuracy:{:.3f}%".format(index, loss, accuracy))
                        if len(params) > 0:
                            print("\n\n\n")
                        self.saver.save(session, self.save_path)

if __name__ == "__main__":
    X = tf.placeholder(tf.float32, shape=(256, 64, 64, 3), name="X")
    Y = tf.placeholder(tf.float32, shape=(256, 5), name="Y")

    N = Network()
    N.model(X).optimizer(Y)