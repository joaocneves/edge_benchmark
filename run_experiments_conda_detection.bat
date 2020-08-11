: OPENVINO R2020.2
: python evaluate_detection_models_openvino.py CPU

: DEVICE
: ENV - Package Manager
: CFS - Compiled from source

: python evaluate_detection_models_tf1.py DEVICE ENV CFS AVX

: TF 1.14
call conda activate tf14
python evaluate_detection_models_tf1.py --device=CPU --env=conda --model_type=Detection

: TF 1.14_cfs
call conda activate tf14_cfs
python evaluate_detection_models_tf1.py --device=CPU --env=conda --avx --model_type=Detection

: TF 1.15
call conda activate tf15
python evaluate_detection_models_tf1.py --device=CPU --env=conda --model_type=Detection

: TF 2.0
call conda activate tf20
python evaluate_detection_models_tf2.py --device=CPU --env=conda --model_type=Detection

: TF 2.0_cfs
call conda activate tf20_cfs
python evaluate_detection_models_tf2.py --device=CPU --env=conda --avx --model_type=Detection

: TF 2.1
call conda activate tf21
python evaluate_detection_models_tf2.py --device=CPU --env=conda --model_type=Detection

: TF 2.1_cfs
call conda activate tf21_cfs
python evaluate_detection_models_tf2.py --device=CPU --env=conda --avx --model_type=Detection

: TF 2.1_cfs
call conda activate tf21_gpu
python evaluate_detection_models_tf2.py --device=GPU --env=conda --avx --model_type=Detection

: TF 2.3
call conda activate tf23
python evaluate_detection_models_tf2.py --device=CPU --env=conda --model_type=Detection

: TF 2.3_cfs
call conda activate tf23_cfs
python evaluate_detection_models_tf2.py --device=CPU --env=conda --avx --model_type=Detection

: TF 2.3_cfs
call conda activate tf23_gpu
python evaluate_detection_models_tf2.py --device=GPU --env=conda --avx --model_type=Detection