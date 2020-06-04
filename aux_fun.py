

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
