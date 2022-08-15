import colorsys
import os
import time
from io import BytesIO
import numpy as np
import torch
import numpy
import cv2
import shutil
import torchvision.utils
from PIL import Image
from torchsummary import summary
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox
from Model_split import cloud
import subprocess
import sys
from sc2bench.transforms.misc import SimpleQuantizer, SimpleDequantizer
from sc2bench.transforms.codec import PillowTensorModule
import copy


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        # print(err.output.decode("utf-8"))
        sys.exit(1)


def HEVC_RGB(outputs,imageid):

    i=0
    #######################################################

    # for i in range(imgNum):
    path_yuv = "D:/HM-master/HM-HM-16.23/workshop/yuv"
    hevc_encoder = r'D:\HM-master\HM-HM-16.23\bin\vs15\msvc-19.16\x86_64\release\TAppEncoder.exe'
    hevc_decoder = r'D:\HM-master\HM-HM-16.23\bin\vs15\msvc-19.16\x86_64\release\TAppDecoder.exe'
    max_image = np.zeros((64,3))
    min_image = np.zeros((64,3))

    p = 0
    outputs = torch.squeeze(outputs, dim=0)
    split_features = outputs.split(3, dim=0)
    #
    for split_feature in split_features:
        max_value = torch.zeros(3)
        min_value = torch.zeros(3)
        normed_feature = torch.zeros(3,160,160)
        p = p +1
        if split_feature.shape[0] == 1:
            split_feature = split_feature.repeat(3,1,1)

        for i in range(3):
            max_value[i] = split_feature[i,:,:].max()
            min_value[i] = split_feature[i,:,:].min()
            max_image[p-1][i] = max_value[i]
            min_image[p-1][i] = min_value[i]
            normed_feature[i,:,:] = (split_feature[i,:,:] - min_value[i]) / (max_value[i] - min_value[i])

        pil_img = to_pil_image(normed_feature)
        # pil_img.save('test.png')
        folder = os.path.exists('C:/Users/宋杰/Desktop/目标检测/yolov4-tiny-pytorch-master/forTestC/PNG_{}_HEVC'.format(str(imageid)))
        if not folder:
            os.makedirs('C:/Users/宋杰/Desktop/目标检测/yolov4-tiny-pytorch-master/forTestC/PNG_{}_HEVC'.format(str(imageid)))
        pil_img.save(("C:/Users/宋杰/Desktop/目标检测/yolov4-tiny-pytorch-master/forTestC/PNG_{}_HEVC/{}.png").format(str(imageid),'test'+str(p)),'png',compress_level=0)

    path_now = "C:/Users/宋杰/Desktop/目标检测/yolov4-tiny-pytorch-master/forTestC/PNG_{}_HEVC".format(str(imageid))

    cmd_ffmpeg = [
        'ffmpeg',
        '-f',
        'image2',
        '-i',
        path_now+'/test%d.png',
        '-pix_fmt',
        'yuv420p',
        '-video_size',
        '160x160',
        path_yuv+'/test{}'.format(imageid)+'.yuv',
    ]
    run_command(cmd_ffmpeg)

    # shutil.rmtree(path_now)
    quality = 30
    #####################
    # -c:v 放-i 前面是解码器   放-i 后面是编码器

    cmd_hard_hevc = [
        'ffmpeg',
        '-f',
        'rawvideo',
        '-video_size',
        '104x104',
        '-i',
        "D:/HM-master/HM-HM-16.23/workshop/yuv/test{}".format(imageid) + '.yuv',
        '-c:v',
        'hevc_nvenc',
        '-crf',
        quality,
        '-preset',
        'ultrafast',
        'D:/HM-master/HM-HM-16.23/workshop/bin/test{}'.format(imageid)+'.265',
    ]

    run_command(cmd_hard_hevc)
    #
    cmd_hevc_decode = [
        'ffmpeg',
        '-c:v',
        'hevc_cuvid',
        '-i',
        'D:/HM-master/HM-HM-16.23/workshop/bin/test{}'.format(imageid)+'.265',
        "D:/HM-master/HM-HM-16.23/workshop/deyuv/test{}".format(imageid) + '.yuv',
    ]

    run_command(cmd_hevc_decode)
    #
    folder = os.path.exists('D:/HM-master/HM-HM-16.23/workshop/depng')
    if not folder:
        os.makedirs('D:/HM-master/HM-HM-16.23/workshop/depng')

    cmd_ffmpeg_decode=[
        'ffmpeg',
        '-f',
        'rawvideo',
        '-video_size',
        '104x104',
        '-i',
        "D:/HM-master/HM-HM-16.23/workshop/deyuv/test{}".format(imageid) + '.yuv',
        '-vframes',
        '64',
        "D:/HM-master/HM-HM-16.23/workshop/depng/test%d.png",
    ]
    run_command(cmd_ffmpeg_decode)

    path_depng = 'D:/HM-master/HM-HM-16.23/workshop/depng'
    dir_depng = os.listdir(path_depng)
    data = np.empty((66,104,104), dtype='float32')
    imgNum = len(dir_depng)
    for id in range(imgNum):
        img = Image.open(path_depng + '/' + "test" + str((id + 1)) + '.png')
        arr = np.asarray(img, dtype='float32')
        # arr = arr.mean(axis=2)
        arr = np.transpose(arr,(2,0,1))
        arr = arr/256
        for i in range(3):
            per_image_max = arr[i,:,:].max()
            per_image_min = arr[i,:,:].min()
            # arr[i,:,:] = (arr[i,:,:] - per_image_min)/(per_image_max - per_image_min)
            arr[i,:,:] = arr[i,:,:]*(max_image[id][i]-min_image[id][i])+min_image[id ][i]
        # data[id,:,:] = arr
        data[3*id:3*(id+1),:,:] = arr

    data = data[0:64,:,:]
    # shutil.rmtree(path_depng)

    print(max_image[1][2],min_image[1][2])
    print(data[5,:,:].max(),data[5,:,:].min())
    return data