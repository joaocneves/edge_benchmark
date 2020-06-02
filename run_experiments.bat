: OPENVINO R2020.2
python evaluate_detection_models_openvino.py CPU

: TF 1.14
call conda activate tf14
python evaluate_detection_models_tf1.py CPU

: TF 1.15
call conda activate tf15
python evaluate_detection_models_tf1.py CPU

: TF 2.0
call conda activate tf20
python evaluate_detection_models_tf2.py CPU

: TF 2.1
call conda activate tf21
python evaluate_detection_models_tf2.py CPU