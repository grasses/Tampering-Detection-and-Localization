#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#
#            God bless me,         no bug!
#                         `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__ = 'Copyright © 2018/08/20, grasses'

import argparse, os
curr_dir = os.path.abspath(os.path.dirname(__file__))

def get_arguments():
    parser = argparse.ArgumentParser(description="Forged Image Detection and Localization with Tensorflow script")
    parser.add_argument("--action", type=str, default="splicing", choices=("generator", "splicing", "train", "test", "download", "clean", "board"), help="Target action")
    parser.add_argument("--conf", type=str, default="code.config", help="Load config file: `code.config` as an example")

    # only for splicing
    parser.add_argument("--source_model", type=str, default="0", help="Splicing source image model id")
    parser.add_argument("--target_model", type=str, default="1", help="Splicing object image model id")

    # only for generator
    parser.add_argument("--run", type=str, default="write", help="Write tfrecord")

    # only for tensorboard
    parser.add_argument("--port", type=int, default=6006, help="Tensorboard port")

    # only for tamper
    parser.add_argument("--type", type=str, default="forged", choices=("forged", "true"), help="Project type")

    # only for download
    parser.add_argument("--name", type=str, default="Dresden", choices=("Dresden",), help="Database name")
    return parser.parse_args()

def main():
    args = get_arguments()
    conf = __import__(args.conf, globals(), locals(), ["Config"]).Config

    if args.action == "splicing":
        '''tampering image with opencv'''
        from code.utils import Splicing
        S = Splicing(conf)
        S.clean()
        S.run2(args.source_model, args.target_model)

    elif args.action == "generator":
        '''build tfrecords'''
        from code.utils import Generator
        G = Generator(conf, debug=True)
        G.write()

    elif args.action == "train":
        '''setup training pipeline'''
        from code.train import main
        main(conf)
    
    elif args.action == "test":
        from code.test import Test
        from code.utils import Splicing
        from code.network import Network
        S = Splicing(conf=conf)
        T = Test(net=Network(conf=conf))

        if args.type == "forged":
            '''test forged images'''
            T.run(S.iterator)
        elif args.type == "true":
            '''test no forged images'''
            T.run(S.file_iterator, is_true_test=True)

    elif args.action == "clean":
        '''clean saved tensorflow model'''
        from code.utils import tools
        check = raw_input("Do you want to clean: {:s}? [Y/N]\n".format(conf.scope_name))
        if check.lower() == "y":
            tools.clean(os.path.join(curr_dir, "code"), conf.scope_name)
            print("=> Clean!")

    elif args.action == "board":
        '''run tensorboard'''
        import subprocess
        path = os.path.join(curr_dir, "code", "board")
        port = args.port
        p = subprocess.Popen(
            ["tensorboard", "--port="+str(port), "--logdir="+str(path)],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        p.stdout.read()

    elif args.action == "download":
        '''download dataset'''
        from code.utils.download import Download
        D = Download(conf, args.name)
        D.run(debug=True)

if __name__ == "__main__":
    main()