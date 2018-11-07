#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/20, grasses'

import tensorflow as tf, numpy as np, os, tools, json, random, cv2
from code.config import Config
curr_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.dirname(curr_path))

class Generator(object):
    def __init__(self, conf=Config, debug=False):
        self._conf = conf
        self.debug = debug

        # saving params
        self.config = {
            "model_label": {},
            "label_model": {},
            "train_list": {},
            "test_list": {},
            "label_path": {},
            "train_label_count": {},
            "test_label_count": {},
            "scope_name": self._conf.scope_name,
            "label_count": self._conf.label_size,

            # dataset
            "dataset_name": conf.dataset_name,

            # network
            "decay_rate": conf.decay_rate,
            "decay_steps": conf.decay_steps,
            "learning_rate": conf.learning_rate,
        }

        self.records_path = "{:s}/{:s}.tfrecords".format(self._conf.output_path, self._conf.scope_name)
        self.model_path = os.path.join(parent_path, "model", self._conf.scope_name)
        tools.rebuild(self.model_path)

    def read_db(self):
        if os.path.exists(os.path.join(self.model_path, "db.json")):
            with open(os.path.join(self.model_path, "db.json"), "r+") as f:
                data = f.read()
                try:
                    self.config = json.loads(data)
                except Exception as e:
                    print(e)
                finally:
                    f.close()
            return True
        return False

    def write_db(self):
        with open(os.path.join(self.model_path, "db.json"), "w+") as f:
            f.write(json.dumps(self.config))
            f.close()
        return self

    def read_directory(self):
        if self.read_db(): return self
        tools.remove_dirty(self._conf.input_path)

        count = 0
        for factory in sorted(os.listdir(self._conf.input_path)):
            tmp_path = os.path.join(self._conf.input_path, factory)
            try:
                for name in sorted(os.listdir(tmp_path)):
                    model = "{:s}_{:s}".format(factory, name)
                    model_dir = "{:s}/{:s}".format(tmp_path, name)
                    if not os.path.isdir(model_dir):
                        os.remove(model_dir)
                        continue
                    self.config["model_label"][model] = count
                    self.config["label_model"][count] = model
                    self.config["test_label_count"][str(count)] = 0
                    self.config["train_label_count"][str(count)] = 0
                    self.config["label_path"][str(count)] = "{:s}/{:s}".format(factory, name)
                    count += 1
            except Exception as e:
                print("=> [error] read_directory() e={:s}".format(e))
        self.config["label_count"] = count

        # push into self.config["train_list"]
        for (label, path) in self.config["label_path"].items():
            tools.remove_dirty(os.path.join(self._conf.input_path, path))

            tmp_list = os.listdir(os.path.join(self._conf.input_path, path))
            self.config["train_list"][str(label)] = tmp_list
            self.config["train_label_count"][str(label)] = len(tmp_list)

        # push into self.config["test_list"]
        for (label, file_list) in self.config["train_list"].items():
            label = str(label)
            self.config["test_list"][label] = []

            if len(file_list) <= self._conf.test_list_size:
                print("=> read_directory() {:s} not enough file for testing".format(self.config["label_model"][int(label)]))
                exit(1)

            for i in range(self._conf.test_list_size):
                fid = random.randint(0, len(self.config["train_list"][label]) - 1)
                self.config["test_list"][label].append(file_list[fid])
                self.config["test_label_count"][label] += 1
        # write into file_list
        self.write_db()
        return self

    '''
    private function
    '''
    def iterator_read(self, is_train=True):
        self.read_directory()
        count = 1
        shape = self._conf.image_size

        while True:
            file_list = self.config["train_list"]
            if not is_train:
                file_list = self.config["test_list"]
            rand_label = str(random.randint(0, self.config["label_count"] - 1))

            # skip file_list[label] = null
            if len(file_list[rand_label]) == 0: continue

            rand_fid = random.randint(0, len(file_list[rand_label]) - 1)
            file_name = file_list[rand_label][rand_fid]
            model_path = self.config["label_path"][rand_label]
            full_name = os.path.join(model_path, file_name)

            if self.debug:
                print("=> iterator_read(), count={:d} filename={:s}".format(count, file_name))

            image = cv2.imread(os.path.join(self._conf.input_path, full_name))
            (w, h) = (int(image.shape[1] / shape[1]), int(image.shape[0] / shape[0]))
            print(full_name, w, h)

            for j in range(h):
                for i in range(w):
                    data = (image[j: j + shape[0], i: i + shape[1], :]).copy()
                    noise = tools.confidence(data)
                    yield(data, int(rand_label), full_name, i * shape[1], j * shape[0], noise)
            count += 1
            del file_list[rand_label][rand_fid]

            # if run all images: break loop
            tmp_count = 0
            for i in range(len(file_list)):
                if len(file_list[str(i)]) == 0:
                    tmp_count += 1
            if tmp_count == self.config["label_count"]: break
        print("=> iterator over")

    '''
    private function
    '''
    def read_and_decode(self):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([self.records_path])
        (_, serialized_example) = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "offset_x": tf.FixedLenFeature([], tf.int64),
                "offset_y": tf.FixedLenFeature([], tf.int64),
                "image_label": tf.FixedLenFeature([], tf.int64),
                "image_noise": tf.FixedLenFeature([], tf.float32),
                "image_data": tf.FixedLenFeature([], tf.string),
                "image_name": tf.FixedLenFeature([], tf.string)
            }
        )
        X = tf.reshape(tf.cast(tf.decode_raw(features["image_data"], tf.uint8), tf.float32), shape=self._conf.image_size)
        Y = tf.cast(features["image_label"], tf.int32)
        (name, image_noise) = (tf.cast(features["image_name"], tf.string), tf.cast(features["image_noise"], tf.float32))
        (offset_x, offset_y) = (tf.cast(features["offset_x"], tf.int32), tf.cast(features["offset_y"], tf.int32))
        return (X, Y, name, offset_x, offset_y, image_noise)


    def crop(self, image, shape, index):
        (x_max, y_max) = int(image.shape[1] / shape[1]), int(image.shape[0] / shape[0])

        if index >= x_max * y_max:
            print(index, x_max, y_max)
            return (None, None, None)

        offset_x, offset_y = int(index % x_max * shape[1]), int(index / x_max * shape[0])
        data = image[offset_y: offset_y + shape[1], offset_x: offset_x + shape[0]].copy()
        return (data, offset_x, offset_y)


    # ========================================public interface============================================== #

    def test_iterator(self):
        count, index = 0, 0
        X, Y = [], []
        for (x, y, name, offset_x, offset_y, noise) in self.iterator_read(is_train=False):
            if index % self._conf.batch_size == 0:
                count += 1
                if index > 0:
                    yield(count, np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32))
                X, Y = [], []

            X.append(x)
            Y.append(y)
            index += 1

    def file_iterator(self, size=None):
        '''
        这个函数主要针对训练，返回训练集中数据；不针对测试，因为不包含图像质量Q、偏移量；
        需要test接口，需要到Splicing.file_iterator
        '''
        self.read_directory()
        shape = self._conf.image_size

        training_list = []
        for (label, names) in self.config["train_list"].items():
            for item in names:
                training_list.append([item, int(label)])
        random.shuffle(training_list)

        count = 0
        step_count = 0
        for (image_name, label) in training_list:
            names = image_name.split("_")
            print("=> {:s}/{:s}/{:s}/{:s}".format(self._conf.input_path, names[0], names[1], image_name))
            image = cv2.imread("{:s}/{:s}/{:s}/{:s}".format(self._conf.input_path, names[0], names[1], image_name))

            (x_max, y_max) = int(image.shape[1] / shape[1]), int(image.shape[0] / shape[0])
            max_patch = x_max * y_max

            (data, offset_x, offset_y) = ([], None, None)
            # 针对图像总数循环
            for index in range(max_patch / self._conf.batch_size + 1):
                (x, y) = [], []
                # 针对batch_size循环
                for i in range(self._conf.batch_size):
                    (_data, _offset_x, _offset_y) = self.crop(image, shape, index)
                    if type(_data) == np.ndarray:
                        data = _data
                    else:
                        print(i, type(_data))
                    x.append(data), y.append(int(label))
                step_count += 1
                yield (count, step_count, np.array(x, dtype=np.float32), np.array(y, dtype=np.int32))
            print("=> count={:d} step_count={:d}".format(count, step_count))
            count += 1
            if size and size > count: return

    '''
    [public funtion] write file into tfrecord, interface from self.iterator_read()
    :param [image_data, image_label, image_name, offset_x, offset_y, image_noise]
    '''
    def write(self):
        self.writer = tf.python_io.TFRecordWriter(self.records_path)
        for (image_data, image_label, image_name, offset_x, offset_y, image_noise) in self.iterator_read():
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "offset_x": tools._int64_feature(offset_x),
                        "offset_y": tools._int64_feature(offset_y),
                        "image_label": tools._int64_feature(image_label),
                        "image_noise": tools._float32_feature(image_noise),
                        "image_data": tools._bytes_feature(image_data.tostring()),
                        "image_name": tools._bytes_feature(str(image_name))
                    }
                )
            )
            self.writer.write(example.SerializeToString())

    '''
    public read interface for tfrecord file, don`t forget to iterator read by tf.Session()
    :param min_after_dequeue    int     queue cahce in memory, a control to shuffle_batch size
    :return [X, Y, name]    name is origin image name
    '''
    def read(self):
        capacity = self._conf.min_after_dequeue * 4
        (_image, _label, _batch_name, _offset_x, _offset_y, _image_noise) = self.read_and_decode()
        (_X, _Y, _name, offset_x, offset_y, noise) = tf.train.shuffle_batch(
            [_image, _label, _batch_name, _offset_x, _offset_y, _image_noise],
            batch_size=self._conf.batch_size,
            capacity=capacity,
            min_after_dequeue=self._conf.min_after_dequeue,
            num_threads=3
        )
        return (_X, _Y, _name, offset_x, offset_y, noise)