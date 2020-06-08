import os
import cv2
import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()


#python C:\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\model_optimizer\extensions\front\tf\ssd_v2_support.json --output="detection_boxes,detection_scores,num_detections"


def load_general_model(model_name, model_dir, sess, graph):

    graph.as_default()
    sess.as_default()

    # restoring the model
    saver = tf.train.import_meta_graph(os.path.join(model_dir, model_name))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    placeholders = [x for x in tf.get_default_graph().get_operations() if x.type == "Placeholder"]
    outputs = [x for x in tf.get_default_graph().get_operations() if x.type == "Softmax"]
    # initializing all variables
    sess.run(tf.global_variables_initializer())

    all_tensors = [t.name for op in graph.get_operations() for t in op.values()]

    input_name = ['input:0']
    output_name = [s for s in all_tensors if 'Logits/Predictions' in s]
    if len(output_name) == 0:
        output_name = [s for s in all_tensors if 'Softmax' in s]
    if len(output_name) == 0:
        output_name = [s for s in all_tensors if 'predictions' in s]
    if len(output_name) == 0:
        output_name = [s for s in all_tensors if 'squeeze' in s]
    # using the model for prediction
    tf_input = graph.get_tensor_by_name(input_name[0])
    tf_output = graph.get_tensor_by_name(output_name[0])

    return tf_input, tf_output

if __name__ == "__main__":


    MODEL_DIR = 'C:/Users/joao_/Downloads/ssd_mobilenet_v1_coco_2018_01_28/'
    MODEL_NAME = 'model.ckpt.meta'

    img = cv2.imread('testimg.jpg')
    img = cv2.resize(img, (300, 300))
    rows = img.shape[0]
    cols = img.shape[1]

    t_s = time.time()
    sess, tf_input, tf_scores, tf_boxes, tf_classes, tf_num_detections = load_detection_model(MODEL_NAME, MODEL_DIR)
    t_e = time.time()
    print(t_e-t_s)

    scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                      feed_dict={tf_input: img[None, ...]})

    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    # Visualize detected bounding boxes.
    for i in range(num_detections):
        classId = int(classes[i])
        score = float(scores[i])
        bbox = [float(v) for v in boxes[i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

            cv2.putText(img, str(classes[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow('TensorFlow MobileNet-SSD', img)
    cv2.waitKey()