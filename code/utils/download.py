#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/11/07, grasses'

import os, requests

class Download():
    def __init__(self, conf, name="Dresden"):
        self._conf = conf
        self._name = name
        self._dataset_root = os.path.join(self._conf.ROOT, "dataset")
        self._dataset_path = os.path.join(self._dataset_root, self._name)

        self.url_path = os.path.join(self._dataset_root, self._name + ".txt")
        if not os.path.exists(self.url_path):
            raise Exception("Dataset url file `{:s}` not exists!".format(self.url_path))
        self.rebuild()

    def rebuild(self):
        try:
            if not os.path.exists(self._dataset_path):
                os.mkdir(self._dataset_path, 0755)
        except Exception as e:
            print("=> error:{:s}".format(str(e)))

    def build_path(self, factory, model):
        try:
            factory_path = os.path.join(self._dataset_path, factory)
            model_path = os.path.join(factory_path, model)
            if not os.path.exists(factory_path):
                os.mkdir(factory_path, 0755)
            if not os.path.exists(model_path):
                os.mkdir(model_path, 0755)
        except Exception as e:
            print("=> error:{:s}".format(str(e)))

    def download(self, url, factory, model, name):
        with open(os.path.join(self._dataset_path, factory, model, name), "wb") as fp:
            response = requests.get(url, stream=True)
            if not response.ok:
                fp.close()
                return False

            for block in response.iter_content(1024):
                if not block:
                    break
                fp.write(block)
            fp.close()
            return True

    def run(self, debug=False):
        errors = []
        with open(self.url_path, "r+") as fp:
            while True:
                line = fp.readline().rstrip("\n")
                name = line.split("/")[-1]
                factory, model = name.split("_")[0], name.split("_")[1]

                self.build_path(factory, model)
                result = self.download(line, factory, model, name)

                if debug: print(line)
                if not result: errors.append(line)
                if not line: break
        return errors