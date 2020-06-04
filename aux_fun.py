

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
    elif 'inception_v2' in model_name:
        return 'Inception-v2'
    elif 'resnet101' in model_name:
        return 'ResNet-101'
    elif 'resnet50' in model_name:
        return 'ResNet-50'
    elif 'mobilenet_v1' in model_name:
        return 'MobileNet-v1'
    elif 'mobilenet_v2' in model_name:
        return 'MobileNet-v2'
    elif 'nas' in model_name:
        return 'NAS'
    else:
        return 'Unknown'
