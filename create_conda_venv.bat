: TF 1.14
call conda create -n tf14 python=3.6 -y
call conda activate tf14
call conda install -y tensorflow==1.14
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
call conda install -y tensorflow==2.0
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