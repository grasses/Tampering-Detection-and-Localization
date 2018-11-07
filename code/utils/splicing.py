#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/07/10, grasses'

import os, sys, cv2, time, json, ntpath, random, threading, numpy as np
from code.config import Config
from code.utils import tools

drawing, mode, flag = False, True, False
times, offset_x, offset_y = 4, 0, 0
offset1, offset2 = 0, 0
_x_, _y_, musk_x, musk_y = 0, 0, 0, 0

curr_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.dirname(curr_path))
(image1, image2, musk_image, musk_shape) = (None, None, None, None)

def get_mask_image(image, musk, _x, _y, with_ground_truth=False):
    '''
    :param (_x, _y) current mouse position
    :param (_x_, _y_) last time mouse position
    :param (musk_x, musk_y) musk image position, maybe negative when mouse negative
    :param (musk_start_x, musk_start_y) / (musk_stop_x, musk_stop_y) crop musk position
    :param (image_start_x, image_start_y) / (image_stop_x, image_stop_y) crop image as background position
    :return: new image with object splicing image
    '''
    global _x_, _y_, musk_x, musk_y

    musk_x, musk_y = _x - _x_ + musk_x, _y - _y_ + musk_y
    _x_, _y_ = _x, _y

    musk_start_x, musk_start_y, musk_stop_x, musk_stop_y = 0, 0, musk.shape[1], musk.shape[0]
    image_start_x, image_start_y, image_stop_x, image_stop_y = musk_x, musk_y, musk.shape[1] + musk_x, musk.shape[0] + musk_y

    shape = list(musk.shape)
    # mouse x overflow
    if _x + shape[1] > image.shape[1]:
        musk_stop_x = image.shape[1] - musk_x
        image_stop_x = image.shape[1]
        shape[1] = image.shape[1] - _x

    # mouse y overflow
    if _y + shape[0] > image.shape[0]:
        musk_stop_y = image.shape[0] - musk_y
        image_stop_y = image.shape[0]

        shape[0] = image.shape[0] - _y

    # mouse x negative
    if musk_x < 0:
        image_start_x = 0
        musk_start_x = -musk_x
        shape[1] = musk.shape[1] + musk_x

    # mouse y negative
    if musk_y < 0:
        image_start_y = 0
        musk_start_y = -musk_y
        shape[0] = musk.shape[0] + musk_y

    shape = tuple(shape)
    _musk = (musk[musk_start_y: musk_stop_y, musk_start_x: musk_stop_x, :]).copy()

    tmp_musk = np.zeros(shape, dtype=np.uint8)
    tmp_musk[_musk == [0, 0, 0]] = 1

    tmp_image = image[image_start_y: image_stop_y, image_start_x: image_stop_x, :]
    tmp_image = tmp_image * tmp_musk[:, :, :]
    tmp_image += _musk

    image[image_start_y: image_stop_y, image_start_x: image_stop_x, :] = tmp_image

    if with_ground_truth:
        '''get ground truth image'''
        image_ground = np.ones(image.shape, dtype=np.uint8) * 255
        tmp_musk = np.ones(shape, dtype=np.uint8) * 255
        tmp_musk[_musk != 0] = 0
        image_ground[image_start_y: image_stop_y, image_start_x: image_stop_x, :] = tmp_musk[:, :, :]
        return (image, image_ground)
    return image

def get_ground_truth2(image, musk, _x, _y):
    shape = musk.shape

    if _x + shape[1] > image.shape[1]: shape[1] = image.shape[1] - _x
    if _y + shape[0] > image.shape[0]: shape[0] = image.shape[0] - _y # shape = [image.shape[0] - _y, shape[1], 3]

    if _x - shape[1] < 0: shape[1] = shape[1] - _x
    if _y - shape[0] < 0: shape = [shape[0] - _y, shape[1], 3]

    image_save = image.copy()
    _musk = musk[0: shape[0], 0: shape[1], :].copy()

    '''get splicing image'''
    tmp_musk = np.zeros(shape, dtype=np.uint8)
    tmp_musk[_musk == 0] = 1

    tmp_image = image[_y: _y + shape[0], _x: _x + shape[1], :]
    tmp_image = tmp_image * tmp_musk[:, :, :]
    tmp_image += _musk
    image_save[_y: _y + shape[0], _x: _x + shape[1], :] = tmp_image

    '''get ground truth image'''
    image_ground = np.ones(image.shape, dtype=np.uint8) * 255
    tmp_musk = np.ones(shape, dtype=np.uint8) * 255
    tmp_musk[_musk != 0] = 0
    image_ground[_y: _y + shape[0], _x: _x + shape[1], :] = tmp_musk[:, :, :]
    return (image_save, image_ground)

def resize_image(image):
    global times
    return cv2.resize(image, (int(image.shape[1] / times), int(image.shape[0] / times)))

def resize_list(_, times):
    data = []
    for i in range(len(_)):
        data.append(int(int(_[i]) * times))
    return data

def find_object(event, x, y, flags, param):
    global image1, musk_image, musk_shape, drawing, mode, point1, point2, point3
    _image = image1.copy()
    (x, y) = resize_list((x, y), times)

    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(_image, point1, 2, (0, 0, 255), times)
        cv2.imshow("find_object", resize_image(_image))

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        point2 = (x, y)
        cv2.rectangle(_image, point1, point2, (0, 0, 255), 2 * times)
        cv2.imshow("find_object", resize_image(_image))

    elif event == cv2.EVENT_LBUTTONUP:
        point3 = (x, y)
        cv2.rectangle(_image, point1, point3, (0, 0, 255), 2 * times)
        cv2.imshow("find_object", resize_image(_image))
        (min_x, min_y) = min(point1[0], point3[0]), min(point1[1], point3[1])
        (width, height) = abs(point1[0] - point3[0]), abs(point1[1] - point3[1])

        if width > 64 and height > 64:
            clips_image = image1[min_y: min_y + height, min_x: min_x + width]
            hsv = cv2.cvtColor(clips_image, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            medianFiltered = cv2.medianBlur(thresholded, 5)
            _, contour_list, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contour_list = sorted(contour_list, key=lambda _: len(_), reverse=True)
            musk_image = np.zeros(clips_image.shape, np.uint8)
            contours_musk = cv2.drawContours(musk_image, contour_list, 0, (255, 255, 255), -1)

            musk_shape = musk_image.shape
            musk_image[contours_musk == 255] = 1
            musk_image = clips_image * musk_image[:, :, :]
            cv2.imshow("object", resize_image(musk_image))

def paste_object(event, x, y, flags, param):
    global image1, image2, musk_image, musk_shape, drawing, point1, point2, point3, offset_x, offset_y, flag
    (x, y) = resize_list((x, y), times)

    if x < 0 or y < 0 or x > image2.shape[1] or y > image2.shape[1]: return
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = False
        point1 = (x, y)
        print("=> mouse down", (x, y))
        if (x - offset_x) <= musk_shape[1] and (y - offset_y) <= musk_shape[0]: flag = True

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        point2 = (x, y)
        _offset_x = offset_x + point2[0] - point1[0]
        _offset_y = offset_y + point2[1] - point1[1]
        _tmp_image = image2.copy()
        cv2.rectangle(_tmp_image, (_offset_x, _offset_y), (_offset_x + musk_shape[1], _offset_y + musk_shape[0]), (0, 0, 255), 2 * times)
        _tmp_image = get_mask_image(_tmp_image, musk_image, _offset_x, _offset_y)
        cv2.imshow("paste_object", resize_image(_tmp_image))

    elif event == cv2.EVENT_LBUTTONUP:
        point3 = (x, y)
        print("=> mouse up", (x, y))
        if flag:
            offset_x += point3[0] - point1[0]
            offset_y += point3[1] - point1[1]
            _tmp_image = image2.copy()
            cv2.rectangle(_tmp_image, (offset_x, offset_y), (offset_x + musk_shape[1], offset_y + musk_shape[0]), (0, 0, 255), 2 * times)
            curr_image = get_mask_image(_tmp_image, musk_image, offset_x, offset_y)
            cv2.imshow("paste_object", resize_image(curr_image))
        flag = False

class Splicing(object):
    def __init__(self, conf=Config):
        self._conf = conf
        self.scope_name = conf.scope_name
        self.input_path = self._conf.input_path
        self.output_path = self._conf.output_path

        self.model_path = os.path.join(parent_path, "model", self._conf.scope_name)
        self.config_path = os.path.join(self.model_path, "db.json")
        self.config = self.load_model_file(self.config_path)
        tools.rebuild(self.model_path)

    def save_model_file(self):
        print("=> {:s}() {:s}".format(sys._getframe().f_code.co_name, self.config_path))
        try:
            with open(self.config_path, "w") as file:
                file.write(json.dumps(self.config))
        except Exception as e:
            print("=> {:s}() error={:s}".format(sys._getframe().f_code.co_name, e))

    def load_model_file(self, path):
        data = []
        try:
            with open(path, "r+") as file:
                data = json.loads(file.read())
        except Exception as e:
            print("=> {:s}() error={:s}".format(sys._getframe().f_code.co_name, e))
        return data

    def get_next_index(self, index, offset, total_size):
        next_index = str((int(index) + offset) % total_size)
        if next_index == index:
            return self.get_next_index(index, offset, total_size)
        return next_index

    def saving(self, _image, path, is_ground_truth=False):
        cv2.imwrite(path, _image)
        if not is_ground_truth:
            self.config["splicing_list"].append(path.split("/")[-1])
            print("=> add into config: {:s}".format(self.config["splicing_list"][-1]))
            print("=> saving splicing image={:s}".format(path))

    def copy_move(self, image1_path, image2_path):
        global image1, image2, musk_image, offset_x, offset_y
        (i, j) = (0, 0)

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        cv2.namedWindow("find_object", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("find_object", find_object)
        cv2.imshow("find_object", resize_image(image1))
        cv2.imshow("paste_object", resize_image(image2))
        cv2.moveWindow("paste_object", image1.shape[1] / times + 5, 0)

        while True:
            k = cv2.waitKey(0)
            if k in [0, 1, 2, 3, 13, ord("q")]:
                cv2.destroyAllWindows()

            if k == 0:
                # key = "up"
                i = -1
                return (i, j, 0, 0)
            elif k == 1:
                # key = "down"
                i = 1
                return (i, j, 0, 0)
            elif k == 2:
                # key = "left"
                j = -1
                return (i, j, 0, 0)
            elif k == 3:
                # key = "right"
                j = 1
                return (i, j, 0, 0)
            elif k > 48 and k < 58:
                return (i, j, 2, k - 49)
            elif k == ord("q"):
                exit()
            elif k == 13:
                # key = "Enter"
                cv2.namedWindow("paste_object", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("paste_object", paste_object)

                curr_image = get_mask_image(image2.copy(), musk_image, offset_x, offset_y)
                cv2.imshow("paste_object", resize_image(curr_image))

                k = cv2.waitKey(0)
                if k in [0, 1, 2, 3, 13, ord("q")]:
                    cv2.destroyAllWindows()

                if k == 0:
                    i = -1
                    return (i, j, 0, 0)
                elif k == 1:
                    i = 1
                    return (i, j, 0, 0)
                elif k == 2:
                    j = -1
                    return (i, j, 0, 0)
                elif k == 3:
                    j = 1
                    return (i, j, 0, 0)
                elif k > 48 and k < 58:
                    return (i, j, 3, k - 49)
                elif k == ord("q"):
                    exit()
                elif k == 13:
                    (curr_image, truth_image) = get_mask_image(image2.copy(), musk_image, offset_x, offset_y, True)
                    base_name = "{:s}_{:s}".format(os.path.splitext(image1_path)[0].split("/")[-1], os.path.splitext(image2_path)[0].split("/")[-1])
                    image_name = "{:s}_{:d}".format(base_name, int(time.time()))

                    threading.Thread(target=self.saving, args=(curr_image, os.path.join(self.model_path, "splicing", image_name + ".bmp"), False)).start()
                    threading.Thread(target=self.saving, args=(truth_image, os.path.join(self.model_path, "ground", image_name + ".jpg"), True)).start()
                    return (1, 1, 1, 0)

    def iterator_build(self, image_list, next_image_list, subpath1, subpath2, model1, model2, cnt=0, max=100):
        global offset1, offset2
        (offset1, offset2) = (0, 0)

        while True:
            if cnt > len(image_list) or cnt > max: break
            image_path1 = os.path.join(subpath1, image_list[offset1])
            image_path2 = os.path.join(subpath2, next_image_list[offset2])

            print(image_path1, image_path2)

            (i, j, _, data) = self.copy_move(image_path1, image_path2)

            if _ == 1:
                if type(self.config["splicing_mark"]) is bool:
                    self.config["splicing_mark"] = {}

                if not self.config["splicing_mark"].has_key("{:s}_{:s}".format(model1, model2)):
                    self.config["splicing_mark"]["{:s}_{:s}".format(model1, model2)] = 0
                self.config["splicing_mark"]["{:s}_{:s}".format(model1, model2)] += 1
                self.save_model_file()
            # 信号2代表"篡改源"更换为k模型
            elif _ == 2:
                threading.Thread(target=self.run2, args=(str(data), model2)).start()
                break
            # 信号3代表"篡改目标"更换为k模型
            elif _ == 3:
                threading.Thread(target=self.run2, args=(model1, str(data))).start()
                break

            cnt += _
            (offset1, offset2) = (offset1 + i + len(image_list)) % len(image_list), (offset2 + j + len(next_image_list)) % len(next_image_list)

    def clean(self):
        self.config = self.load_model_file(self.config_path)
        base_path = os.path.join(parent_path, "model", self._conf.scope_name)
        ground_list = os.listdir(os.path.join(base_path, "ground"))
        splicing_list = os.listdir(os.path.join(base_path, "splicing"))

        print("=> I find file list:\n")
        for item in splicing_list:
            print("[{:s}]".format(item))

        choice = raw_input("=> Do you want to remove all? [y/n]\n")
        if choice.lower() == "y":
            print("=> removing files...")
            for item in splicing_list:
                os.unlink(os.path.join(base_path, "splicing", item))
            for item in ground_list:
                os.unlink(os.path.join(base_path, "ground", item))
            self.config["splicing_list"] = []
            self.config["splicing_mark"] = False
            self.save_model_file()

    def single_read(self, splice_path, ground_path, model1=None):
        shape = self._conf.image_size
        splice = self._conf.splicing["splice"]
        stride = self._conf.splicing["stride"]
        batch_size = self._conf.batch_size

        splice_image = cv2.imread(splice_path)
        (width, height) = (splice_image.shape[1], splice_image.shape[0])

        if not model1:
            name = ntpath.basename(ground_path)
            model1 = self.config["model_label"]["{:s}_{:s}".format(name.split("_")[0], name.split("_")[1])]
            model2 = self.config["model_label"]["{:s}_{:s}".format(name.split("_")[4], name.split("_")[5])]
            print("=> model1={:d} model2={:d}".format(model1, model2))
            ground_image = cv2.imread(ground_path)
        else:
            name = ntpath.basename(splice_path)
            model2 = model1
            ground_image = np.ones(splice_image.shape) * 255

        x_max_index, y_max_index = int(width / stride - 1 + int(stride / shape[1])), int(height / stride - 1 + int(stride / shape[1]))
        total_crop = x_max_index * y_max_index

        ground_image = ground_image[:, :, 0].copy()
        ground_image[ground_image < 1] = 1
        ground_image[ground_image > 1] = 0

        count, X, Y, tamper_rate, offset_x, offset_y, confidence = 0, None, None, None, None, None, None
        flag = total_crop / batch_size if total_crop % batch_size == 0 else total_crop / batch_size + 1
        for index in range(total_crop):
            _index = index % batch_size
            if index > 0 and _index == 0:
                count += 1
                yield (count, flag, name.split(".")[0], X, Y, confidence, tamper_rate, splice_image.shape, offset_x, offset_y, width, height)

            # init value slot
            if _index == 0:
                X = np.zeros([batch_size, shape[0], shape[1], 3], dtype=np.uint8)
                Y = np.ones([batch_size], dtype=np.int32) * int(model2)
                offset_x = np.zeros([batch_size], dtype=np.int32)
                offset_y = np.zeros([batch_size], dtype=np.int32)
                tamper_rate = np.zeros([batch_size], dtype=np.float32)
                confidence = np.zeros([batch_size], dtype=np.float32)

            # get slide windows offset index and offset pixel
            x_index, y_index = int(index % x_max_index), int(index / x_max_index)
            (_offset_x, _offset_y) = (stride * (x_index), stride * (y_index))

            # extract slide window image, splice rate in a slide window

            crop_imgage = (splice_image[_offset_y: _offset_y + shape[0], _offset_x: _offset_x + shape[1], :]).copy()
            crop_ground = ground_image[_offset_y: _offset_y + shape[0], _offset_x: _offset_x + shape[1]].copy()
            tamper_rate[_index] = float(np.sum(crop_ground) / (shape[0] * shape[1]))
            confidence[_index] = tools.confidence(crop_imgage)

            X[_index] = crop_imgage
            offset_x[_index] = _offset_x
            offset_y[_index] = _offset_y
            if tamper_rate[_index] > splice: Y[_index] = int(model1)

            # if not satisfy batch_size windows, feed last window in remaining blocks data
        if total_crop % batch_size != 0:
            for i in range(batch_size - total_crop % batch_size):
                _index = total_crop % batch_size + i
                X[_index] = X[_index - 1]
                Y[_index] = Y[_index - 1]
                tamper_rate[_index] = tamper_rate[_index - 1]
                offset_x[_index] = offset_x[_index - 1]
                offset_y[_index] = offset_y[_index - 1]
                confidence[_index] = confidence[_index - 1]
            yield (count + 1, flag, name.split(".")[0], X, Y, confidence, tamper_rate, splice_image.shape, offset_x, offset_y, width, height)

    def iterator(self):
        for item in self.config["splicing_list"]:
            path1 = os.path.join(self.model_path, "splicing", item)
            path2 = os.path.join(self.model_path, "ground", item.split(".")[0] + ".jpg")
            print(path1, path2)
            for (count, flag, name, X, Y, Q, tamper_rate, shape, offset_x, offset_y, width, height) in self.single_read(path1, path2):
                yield(count, flag, name, X, Y, Q, tamper_rate, shape, offset_x, offset_y, width, height)

    def file_iterator(self):
        self.config = self.load_model_file(self.config_path)

        for (label, files) in self.config["test_list"].items():
            for name in files:
                name_split = name.split("_")
                path1 = os.path.join(self._conf.input_path, name_split[0], name_split[1], name)
                path2 = None

                for (count, flag, name, X, Y, Q, tamper_rate, shape, offset_x, offset_y, width, height) in self.single_read(path1, path2, label):
                    yield (count, flag, name, X, Y, Q, tamper_rate, shape, offset_x, offset_y, width, height)

    def run(self):
        self.config = self.load_model_file(self.config_path) if not self.config else self.config
        keys = self.config["test_list"].keys()
        random.shuffle(keys)
        if not self.config.has_key("splicing_list"): self.config["splicing_list"] = []
        if not self.config.has_key("splicing_mark"): self.config["splicing_mark"] = {}

        offset = 1
        for index in keys:
            image_list = self.config["test_list"][index]
            next_index = self.get_next_index(index, offset, len(self.config["label_path"]))
            subpath1 = os.path.join(self.input_path, self.config["label_path"][index])
            subpath2 = os.path.join(self.input_path, self.config["label_path"][next_index])
            self.iterator_build(image_list, self.config["test_list"][next_index], subpath1, subpath2, index, next_index)
            self.save_model_file()

    def run2(self, model1, model2, max_size=10):
        self.config = self.load_model_file(self.config_path) if not self.config else self.config
        print(str(model1), str(model2))
        (model1, model2, keys) = (str(model1), str(model2), self.config["test_list"].keys())

        if not model1 in keys or not model2 in keys: exit(1)
        if not self.config.has_key("splicing_list"): self.config["splicing_list"] = []
        if not self.config.has_key("splicing_mark"): self.config["splicing_mark"] = {}

        image_list1 = sorted(self.config["test_list"][model1])
        image_list2 = sorted(self.config["test_list"][model2])
        subpath1 = os.path.join(self.input_path, self.config["label_path"][model1])
        subpath2 = os.path.join(self.input_path, self.config["label_path"][model2])
        self.iterator_build(image_list1, image_list2, subpath1, subpath2, model1, model2, max=max_size)
        self.save_model_file()

if __name__ == "__main__":
    pass






