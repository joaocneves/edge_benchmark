: TF 1.14
call conda create -n tf14 python=3.6 -y
call conda activate tf14
call conda install -y tensorflow==1.14
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 1.14 (compiled from source)
call conda create -n tf14_cfs python=3.6 -y
call conda activate tf14_cfs
call pip install C:\\tmp\\tensorflow_pkg\\tensorflow-1.14.1-cp36-cp36m-win_amd64.whl
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 1.15
call conda create -n tf15 python=3.6 -y
call conda activate tf15
call conda install -y tensorflow==1.15
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.0
call conda create -n tf20 python=3.6 -y
call conda activate tf20
call pip install C:\\tmp\\tensorflow_pkg\\tensorflow-2.0.2-cp36-cp36m-win_amd64.whl
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.1
call conda create -n tf21 python=3.6 -y
call conda activate tf21
call conda install -y tensorflow==2.1
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.1
call conda create -n tf21_cfs python=3.6 -y
call conda activate tf21_cfs
call conda install -y C:\\tmp\\tensorflow_pkg\\tensorflow-2.1.1-cp36-cp36m-win_amd64.whl
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.1
call conda create -n tf21_gpu python=3.6 -y
call conda activate tf21_gpu
call conda install -y tensorflow-gpu==2.1
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.3
call conda create -n tf23 python=3.6 -y
call conda activate tf23
call conda install -y tensorflow==2.3
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.3
call conda create -n tf23_cfs python=3.6 -y
call conda activate tf23_cfs
call conda install -y C:\\tmp\\tensorflow_pkg\\tensorflow-2.3.0-cp36-cp36m-win_amd64.whl
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI

: TF 2.3
call conda create -n tf23_gpu python=3.6 -y
call conda activate tf23_gpu
call conda install -y tensorflow-gpu==2.3
call conda install -c conda-forge -y opencv
call conda install -y py-cpuinfo
call conda install -y pandas
call pip install WMI