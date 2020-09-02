import os
from lib.aux_fun import  tf_network_name
import argparse

def get_ckpt_name(model_path):

    files = os.listdir(model_path)

    matching = [s for s in files if s.endswith('ckpt')]
    if len(matching):
        return matching[0]

    matching = [s for s in files if "ckpt.data" in s]
    if len(matching):
        return matching[0]

    return ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_slim_path', type=str, default='C:/Users/joao_/Repos/', help='the path where the tensorflow models repo is located')
    args = parser.parse_args()

    MODELS_DIR = 'general_models'
    MODEL_OPTIMIZER_PATH = 'C:/Program Files (x86)/IntelSWTools/openvino_2020.2.117/deployment_tools/model_optimizer/'

    models = [name for name in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, name))]

    for model in models:

        model_path = os.path.join(MODELS_DIR, model,'')
        network_name = tf_network_name(model)
        print(model)
        print(network_name)


    for model in models:

        model_path = os.path.join(MODELS_DIR, model,'')
        network_name = tf_network_name(model)
        ckpt_name = get_ckpt_name(model_path)
        pb_name = 'frozen_inference_graph.pb'

        'An offset for the labels in the dataset. This flag is primarily used to '
        'evaluate the VGG and ResNet architectures which do not use a background '
        'class for the ImageNet dataset.'
        if(network_name.startswith('vgg') or network_name.startswith('resnet_v1')):
            labels_offset = 1
        else:
            labels_offset = 0

        cmd = 'python {0}tf_models/research/slim/export_inference_graph.py ' \
            '--model_name {1} ' \
            '--output_file {2} ' \
            '--labels_offset {3}' \
            .format(args.tf_slim_path, network_name, os.path.join(model_path, pb_name), labels_offset)

        print(cmd)
        os.system(cmd)

        cmd = 'python "{0}mo_tf.py" --input_model {1}{2} ' \
              '--input_checkpoint {1}{3} ' \
              '-b 1 --output_dir {1}' \
            .format(MODEL_OPTIMIZER_PATH, model_path, pb_name, ckpt_name)

        if model.startswith('mobilenet') or model.startswith('nasnet'):
            cmd = 'python "{0}mo_tf.py" --input_meta_graph {1}model.ckpt.meta  ' \
                  '-b 1 --output_dir {1}' \
                .format(MODEL_OPTIMIZER_PATH, model_path)

        print(cmd)
        os.system(cmd)
        print('ok')