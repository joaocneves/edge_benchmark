import numpy as np
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def tf_network_name(model_name):

    networks =  {'vgg_16_2016_08_28': 'vgg_16',
                 'vgg_19_2016_08_28': 'vgg_19',
                 'inception_resnet_v2_2016_08_30': 'inception_resnet_v2',
                 'inception_v1_2016_08_28': 'inception_v1',
                 'inception_v2_2016_08_28': 'inception_v2',
                 'inception_v3_2016_08_28': 'inception_v3',
                 'inception_v4_2016_09_09': 'inception_v4',
                 'resnet_v1_50_2016_08_28': 'resnet_v1_50',
                 'resnet_v1_101_2016_08_28': 'resnet_v1_101',
                 'resnet_v1_152_2016_08_28': 'resnet_v1_152',
                 'resnet_v2_50_2017_04_14': 'resnet_v2_50',
                 'resnet_v2_101_2017_04_14': 'resnet_v2_101',
                 'resnet_v2_152_2017_04_14': 'resnet_v2_152',
                 'mobilenet_v1_1.0_224': 'mobilenet_v1',
                 'mobilenet_v1_0.5_160': 'mobilenet_v1_050',
                 'mobilenet_v2_1.0_224': 'mobilenet_v2',
                 'mobilenet_v2_1.4_224': 'mobilenet_v2_140',
                 'nasnet-a_large_04_10_2017': 'nasnet_large',
                 'nasnet-a_mobile_04_10_2017': 'nasnet_mobile'
                 }

    return networks[model_name]


def model_type(model_name):

    if model_name.startswith('ssd'):
        return 'SSD'
    elif model_name.startswith('rfcn'):
        return 'R-FCN'
    elif model_name.startswith('faster'):
        return 'Faster R-CNN'
    elif model_name.startswith('mask'):
        return 'Mask R-CNN'
    else:
        return 'Unknown'

def architecture_type(model_name):

    if 'inception_resnet_v2' in model_name:
        return 'Inception-ResNet-v2'
    elif 'inception_v1' in model_name:
        return 'Inception-v1'
    elif 'inception_v2' in model_name:
        return 'Inception-v2'
    elif 'inception_v3' in model_name:
        return 'Inception-v3'
    elif 'inception_v4' in model_name:
        return 'Inception-v4'
    elif 'resnet_v1_50' in model_name:
        return 'ResNet-50-v1'
    elif 'resnet_v1_101' in model_name:
        return 'ResNet-101-v1'
    elif 'resnet_v1_152' in model_name:
        return 'ResNet-152-v1'
    elif 'resnet_v2_50' in model_name:
        return 'ResNet-50-v2'
    elif 'resnet_v2_101' in model_name:
        return 'ResNet-101-v2'
    elif 'resnet_v2_152' in model_name:
        return 'ResNet-152-v2'
    elif 'mobilenet_v1' in model_name:
        return 'MobileNet-v1'
    elif 'mobilenet_v2' in model_name:
        return 'MobileNet-v2'
    elif 'nas' in model_name:
        return 'NAS'
    elif 'vgg_16' in model_name:
        return 'VGG-16'
    elif 'vgg_19' in model_name:
        return 'VGG-19'
    else:
        return 'Unknown'
