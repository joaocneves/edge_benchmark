import os
import sys
import cpuinfo
import argparse
import numpy as np
import pandas as pd
import statistics as st
from lib.aux_fun import model_type as model_type_fun
from lib.aux_fun import architecture_type as architecture_type_fun
from lib.aux_fun import backbone_type as backbone_type_fun

def process_model(model, df, args):

    inference_data = 'test_img.jpeg'
    model_path = os.path.join(args.models_dir, model)
    model_name = 'frozen_inference_graph.xml'
    if not os.path.exists(os.path.join(model_path, model_name)):
        model_name = 'model.ckpt.xml'

    if args.model_type=='Detection':
        arch_type = model_type_fun(model) + ' (' + backbone_type_fun(model) + ')'
    else:
        arch_type = architecture_type_fun(model)

    print('Processing ' + model)

    cmd = 'benchmark_app\\benchmark_app.exe -i detection_images\\{0} -api sync -d {1} -niter {2} -m {3}' \
        .format(inference_data, args.device, args.inference_iters, os.path.join(model_path, model_name))

    print(cmd)
    os.system(cmd)
    consumed_time = np.loadtxt('latencies.txt')

    if args.device == 'MYRIAD':
        args.device = args.device_name

    df = df.append({'Model': model, 'Model Type': args.model_type, 'Architecture': arch_type,
                    'Flops': 0, 'Framework': args.framework, 'Device Name': args.device_name,
                    'Device': args.device, 'Environment': args.env, 'AVX': args.avx,
                    'Average Time': st.mean(consumed_time), 'Std Time': st.stdev(consumed_time)}, ignore_index=True)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='CPU', help='hardware device to run model')
    parser.add_argument('--env', type=str, default='conda', help='python environment')
    parser.add_argument('--framework', type=str, default='Openvino 2020.2',
                        help='deep learning framework to run the models')
    parser.add_argument('--avx', action='store_true', help='the framework uses AVX?')
    parser.add_argument('--model_type', type=str, default='Detection', help='the task to which the model was built to')
    parser.add_argument('--images_dir', type=str, default='detection_images/',
                        help='the path containing the images to process')
    parser.add_argument('--models_dir', type=str, default='detection_models/',
                        help='the path containing the models to process')
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
    else:
        args.device_name = args.device + ' X'

    models = [name for name in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, name))]
    images = [name for name in os.listdir(args.images_dir) if name.endswith('png') or name.endswith('jpg')]

    outfile = 'stats_{0}_{1}_{2}_{3}_{4}.csv'.format(args.model_type, args.env, args.framework, suffix_AVX,
                                                     args.device_name)

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
        df = process_model(model, df, args)

    df.to_csv(outfile)
