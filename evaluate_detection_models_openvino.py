import os
import sys
import cv2
import wmi
import time
import cpuinfo
import numpy as np
import pandas as pd
import statistics as st



if __name__ == '__main__':

    # ---------------- PARAMS -------------------- #

    DEVICE = 'CPU'
    FRAMEWORK = 'Openvino 2020.2'

    if len(sys.argv) > 1:
        DEVICE = sys.argv[1]

    if len(sys.argv) > 2:
        FRAMEWORK = sys.argv[2]

    MODELS_DIR = 'detection_models/'
    IMAGES_DIR = 'detection_images/'

    if DEVICE == 'CPU':
        inference_device = cpuinfo.get_cpu_info()['brand']
    elif DEVICE == 'GPU':
        computer = wmi.WMI()
        gpu_info = computer.Win32_VideoController()[0]
        inference_device = gpu_info.Name


    models = [name for name in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, name))]
    images = [name for name in os.listdir(IMAGES_DIR) if name.endswith('png') or name.endswith('jpg')]

    inference_data = 'test_img.jpeg'
    inference_iters = 10
    outfile = 'stats_{0}_{1}.csv'.format(FRAMEWORK, inference_device)


    df = pd.DataFrame({'Model': [],
                       'Flops': [],
                       'Framework': [],
                       'Device': [],
                       'Average Time': [],
                       'Std Time': []
                       })


    # g = tf.Graph()
    # run_meta = tf.RunMetadata()
    # with g.as_default():
    #     A = tf.Variable(initial_value=tf.zeros([25, 16]))
    #     B = tf.Variable(initial_value=tf.zeros([16, 9]))
    #     C = tf.matmul(A, B) # shape=[25,9]
    #
    #     opts = tf.profiler.ProfileOptionBuilder.float_operation()
    #     flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    #     if flops is not None:
    #         print('Flops should be ~', 2*25*16*9)
    #         print('TF stats gives', flops.total_float_ops)

    for model in models[13:15]:

        model_path = os.path.join(MODELS_DIR, model)
        model_name = 'frozen_inference_graph.xml'

        print('Processing ' + model)



        cmd = 'benchmark_app\\benchmark_app.exe -i detection_images\\{0} -api sync -d {1} -niter {2} -m {3}'\
            .format(inference_data, DEVICE, inference_iters, os.path.join(model_path, model_name))

        print(cmd)
        os.system(cmd)
        consumed_time = np.loadtxt('latencies.txt')

        df = df.append({'Model': model, 'Flops': 0, 'Framework': FRAMEWORK, 'Device': inference_device,
                   'Average Time': st.mean(consumed_time), 'Std Time': st.stdev(consumed_time)}, ignore_index=True)



    df.to_csv(outfile)