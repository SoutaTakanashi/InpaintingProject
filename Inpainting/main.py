#!/usr/bin/python 
##-*- coding : utf-8 -*-
import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile #Python的关于文件操作包
from src.config import Config
from src.edge_connect import EdgeConnect


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device       查看是否有可用gpu
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed 随机种子是为了让每次跑出来的结果一样
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)#为所有GPU设置随机种子
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = EdgeConnect(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\n训练中...\n')
        model.train()

    # model test
    else:
        print('\n以下为结果输出~\n')
        model.test()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    #print(args.path)
    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # load config file
    config = Config(config_path)
    print('配置文件位置 =',args.path,'config.yml')
    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    else :
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 0

        if args.input is not None:#分别对Edge_connect的TEST_FLIST等属性进行赋值
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
#    elif mode == 3:
 #       config.MODE = 3
  #      config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
