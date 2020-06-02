import os
import sys
import cv2
import wmi
import time
import cpuinfo
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()
import statistics as st
from detection_model_lib import load_detection_model




if __name__ == '__main__':

    # ---------------- PARAMS -------------------- #

    DEVICE = 'CPU'
    FRAMEWORK = 'Tensorflow ' + tf.__version__

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
        model_name = 'model.ckpt.meta'

        sess = tf.Session()
        graph = tf.get_default_graph()
        print('Processing ' + model)

        tf_input, tf_scores, tf_boxes, tf_classes, tf_num_detections =\
            load_detection_model(model_name, model_path, sess, graph)

        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.all_variables()]))

        #all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('Model {} needs {} FLOPS after freezing'.format(model, flops.total_float_ops))


        consumed_time = []
        # first inference takes longer than others
        # this allows to discard it from results
        scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                          feed_dict={tf_input: np.zeros((1,300, 300, 3))})

        image_data = cv2.imread(os.path.join(IMAGES_DIR, inference_data))
        # image_data = cv2.resize(image_data, (300, 300))

        for it in range(inference_iters):

            t_s = time.time()
            scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                              feed_dict={tf_input: image_data[None, ...]})
            t_e = time.time()

            consumed_time.append(1000*(t_e - t_s))


        df = df.append({'Model': model, 'Flops': flops.total_float_ops, 'Framework': FRAMEWORK, 'Device': inference_device,
                   'Average Time': st.mean(consumed_time), 'Std Time': st.stdev(consumed_time)}, ignore_index=True)


        sess.close()
        tf.reset_default_graph()


    df.to_csv(outfile)