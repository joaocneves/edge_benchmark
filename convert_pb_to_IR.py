import os

MODELS_DIR = 'detection_models'
MODEL_OPTIMIZER_PATH = 'C:/Program Files (x86)/IntelSWTools/openvino_2020.2.117/deployment_tools/model_optimizer/'

models = [name for name in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, name))]

MODEL_OPTIMIZER_PATH
for model in models:

    model_path = os.path.join(MODELS_DIR, model,'')

    if model.startswith('ssd'):

        cmd = 'python "{0}mo_tf.py" --input_model {1}{2} ' \
              '--tensorflow_object_detection_api_pipeline_config {1}pipeline.config ' \
              '--transformations_config "{0}extensions/front/tf/ssd_v2_support.json" ' \
              '--output detection_boxes,detection_scores,num_detections ' \
              '--output_dir {1}' \
            .format(MODEL_OPTIMIZER_PATH, model_path, 'frozen_inference_graph.pb' )

    elif model.startswith('rfcn'):

        cmd = 'python "{0}mo_tf.py" --input_model {1}{2} ' \
              '--tensorflow_object_detection_api_pipeline_config {1}pipeline.config ' \
              '--transformations_config "{0}extensions/front/tf/rfcn_support.json" ' \
              '--output detection_boxes,detection_scores,num_detections ' \
              '--output_dir {1}' \
            .format(MODEL_OPTIMIZER_PATH, model_path, 'frozen_inference_graph.pb')

    elif model.startswith('faster'):

        cmd = 'python "{0}mo_tf.py" --input_model {1}{2} ' \
              '--tensorflow_object_detection_api_pipeline_config {1}pipeline.config ' \
              '--transformations_config "{0}extensions/front/tf/faster_rcnn_support.json" ' \
              '--output detection_boxes,detection_scores,num_detections ' \
              '--output_dir {1}' \
            .format(MODEL_OPTIMIZER_PATH, model_path, 'frozen_inference_graph.pb')

    elif model.startswith('mask'):

        cmd = 'python "{0}mo_tf.py" --input_model {1}{2} ' \
              '--tensorflow_object_detection_api_pipeline_config {1}pipeline.config ' \
              '--transformations_config "{0}extensions/front/tf/mask_rcnn_support.json" ' \
              '--output detection_boxes,detection_scores,num_detections ' \
              '--output_dir {1}' \
            .format(MODEL_OPTIMIZER_PATH, model_path, 'frozen_inference_graph.pb')

    else:

        print('Unrecognized model')
        cmd = 'Unrecognized model'

    print(cmd)
    os.system(cmd)
