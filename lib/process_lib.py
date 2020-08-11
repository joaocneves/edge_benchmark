import os
import cv2
import time
import numpy as np
import statistics as st
from lib.detection_model_lib import load_detection_model
from lib.general_model_lib import load_general_model
from lib.aux_fun import model_type as model_type_fun
from lib.aux_fun import architecture_type as architecture_type_fun
from lib.aux_fun import backbone_type as backbone_type_fun

def process_model(model, df, args, tf):

    inference_data = 'test_img.jpeg'
    model_path = os.path.join(args.models_dir, model)
    model_name = 'model.ckpt.meta'

    if args.model_type=='Detection':
        arch_type = model_type_fun(model) + ' (' + backbone_type_fun(model) + ')'
    else:
        arch_type = architecture_type_fun(model)

    sess = tf.Session()
    graph = tf.get_default_graph()
    print('Processing ' + model)

    if args.model_type=='Detection':
        tf_input, tf_scores, tf_boxes, tf_classes, tf_num_detections = \
            load_detection_model(model_name, model_path, sess, graph)
    else:
        tf_input, tf_output = \
            load_general_model(model_name, model_path, sess, graph)

    input_shape = tf_input.shape[1:]

    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.all_variables()]))

    # all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('Model {} needs {} FLOPS after freezing'.format(model, flops.total_float_ops))

    consumed_time = []
    # first inference takes longer than others
    # this allows to discard it from results
    if args.model_type=='Detection':
        scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                          feed_dict={tf_input: np.zeros((1, 300, 300, input_shape[2]))})
        image_data = cv2.imread(os.path.join(args.images_dir, inference_data))
        image_data = cv2.resize(image_data, (300, 300))
    else:
        scores = sess.run(tf_output,
                          feed_dict={tf_input: np.zeros((1, input_shape[0], input_shape[1], input_shape[2]))})
        image_data = cv2.imread(os.path.join(args.images_dir, inference_data))
        image_data = cv2.resize(image_data, (input_shape[0], input_shape[1]))

    for it in range(args.inference_iters):

        t_s = time.time()
        if args.model_type=='Detection':
            scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                              feed_dict={tf_input: image_data[None, ...]})
        else:
            scores = sess.run(tf_output, feed_dict={tf_input: image_data[None, ...]})
        t_e = time.time()

        consumed_time.append(1000 * (t_e - t_s))

    df = df.append({'Model': model, 'Model Type': args.model_type, 'Architecture': arch_type,
                    'Flops': flops.total_float_ops, 'Framework': args.framework, 'Device Name': args.device_name,
                    'Device': args.device, 'Environment': args.env, 'AVX': args.avx,
                    'Average Time': st.mean(consumed_time), 'Std Time': st.stdev(consumed_time)}, ignore_index=True)

    sess.close()
    tf.reset_default_graph()

    return df
