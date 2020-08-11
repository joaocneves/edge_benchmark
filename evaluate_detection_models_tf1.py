import os
import wmi
import cpuinfo
import argparse
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('INFO')
#import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()
from lib.process_lib import process_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='CPU', help='hardware device to run model')
    parser.add_argument('--env', type=str, default='conda', help='python environment')
    parser.add_argument('--framework', type=str, default='Tensorflow ' + tf.__version__, help='deep learning framework to run the models')
    parser.add_argument('--avx', action='store_true', help='the framework uses AVX?')
    parser.add_argument('--model_type', type=str, default='Detection', help='the task to which the model was built to')
    parser.add_argument('--images_dir', type=str, default='detection_images/', help='the path containing the images to process')
    parser.add_argument('--models_dir', type=str, default='detection_models/', help='the path containing the models to process')
    parser.add_argument('--inference_iters', type=int, default=5, help='the number of inference iterations to perform')
    parser.add_argument('--device_name', type=str, default='', help='the name of the device to run model')

    args = parser.parse_args()
    # ---------------- PARAMS -------------------- #

    if args.model_type == 'Detection':
        args.models_dir = 'detection_models/'
    else:
        args.models_dir = 'general_models/'

    if args.avx:
        suffix_AVX = 'AVX'
    else:
        suffix_AVX = 'NOAVX'

    aux = cpuinfo.get_cpu_info()
    if args.device == 'CPU':
        if 'brand' in aux.keys():
            args.device_name = cpuinfo.get_cpu_info()['brand']
        else:
            args.device_name = cpuinfo.get_cpu_info()['brand_raw']
    elif args.device == 'GPU':
        computer = wmi.WMI()
        gpu_info = computer.Win32_VideoController()[0]
        args.device_name = gpu_info.Name

    models = [name for name in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, name))]
    images = [name for name in os.listdir(args.images_dir) if name.endswith('png') or name.endswith('jpg')]

    outfile = 'stats_{0}_{1}_{2}_{3}_{4}.csv'.format(args.model_type, args.env, args.framework, suffix_AVX, args.device_name)

    df = pd.DataFrame({'Model': [],
                       'Model Type': [],
                       'Architecture': [],
                       'Flops': [],
                       'Framework': [],
                       'Device': [],
                       'Device Name': [],
                       'Environment': [],
                       'AVX': [],
                       'Average Time': [],
                       'Std Time': []
                       })

    for model in models:

        df = process_model(model, df, args, tf)


    df.to_csv(outfile)