## Install Dependencies

create_conda_venv.bat


## Download and Create Models

### 1)  Download detection models (.pb representation)

python download_models.py --model_type=Detection
python download_models.py --model_type=Classification

### 2) Generate .pb representation of the models

git clone https://github.com/tensorflow/models.git tf_models

python convert_ckpt_to_pb.py --tf_slim_path=E:\\Repos\\

it may be necessary to change lines 93 and 98

filename, _ = urllib.request.urlretrieve(synset_url)
filename = 'E:\\Repos\\tf_models\\research\\slim\\datasets\\imagenet_lsvrc_2015_synsets.txt'

filename, _ = urllib.request.urlretrieve(synset_to_human_url)
filename = 'E:\\Repos\\tf_models\\research\\slim\\datasets\\imagenet_metadata.txt'

### 3) Generate openvino compatible models (IR representation)

python convert_pb_to_IR.py

## Create Virtual Environments

create_python_venv.bat

## Run Experiments

run_experiments.bat
