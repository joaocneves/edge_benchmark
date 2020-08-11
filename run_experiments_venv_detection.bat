: python evaluate_detection_models_tf1.py DEVICE ENV CFS AVX DETECTION_MODELS FRAMEWORK

: TF 1.14
call venvs\tf14\Scripts\activate
python evaluate_detection_models_tf1.py --device=CPU --env=venv --model_type=Detection

: TF 1.14
call venvs\tf14_cfs\Scripts\activate
python evaluate_detection_models_tf1.py --device=CPU --env=venv --avx --model_type=Detection

: TF 1.15
call venvs\tf15\Scripts\activate
python evaluate_detection_models_tf1.py --device=CPU --env=venv --model_type=Detection

: TF 2.0
call venvs\tf20\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --model_type=Detection

: TF 2.0
call venvs\tf20_cfs\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --avx --model_type=Detection

: TF 2.1
call venvs\tf21\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --model_type=Detection

: TF 2.1
call venvs\tf21_cfs\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --avx --model_type=Detection

: TF 2.1
call venvs\tf21_gpu\Scripts\activate
python evaluate_detection_models_tf2.py --device=GPU --env=venv --avx --model_type=Detection

: TF 2.3
call venvs\tf23\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --model_type=Detection

: TF 2.3
call venvs\tf23_cfs\Scripts\activate
python evaluate_detection_models_tf2.py --device=CPU --env=venv --avx --model_type=Detection

: TF 2.3
call venvs\tf23_gpu\Scripts\activate
python evaluate_detection_models_tf2.py --device=GPU --env=venv --avx --model_type=Detection