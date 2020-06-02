## Install Dependencies

create_conda_venv.bat


## Download and Create Models

1)  Download detection models (.pb representation)

python download_detection_models.py

2) Generate openvino compatible models (IR representation)

python convert_pb_to_IR.py

## Run Experiments

run_experiments.bat
