: TF 1.14
: pip install virtualenv
virtualenv venvs\tf14
call venvs\tf14\Scripts\activate
pip install tensorflow==1.14
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 1.14_cfs
: virtualenv venvs\tf14_cfs
: call venvs\tf14_cfs\Scripts\activate
: pip install C:\\tmp\\tensorflow_pkg\\tensorflow-1.14.1-cp36-cp36m-win_amd64.whl
: pip install opencv-python
: pip install py-cpuinfo
: pip install pandas
: pip install WMI

: TF 1.15
virtualenv venvs\tf15
call venvs\tf15\Scripts\activate
pip install tensorflow==1.15.2
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 2.0
virtualenv venvs\tf20
call venvs\tf20\Scripts\activate
pip install tensorflow==2.0.2
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 2.0
: virtualenv venvs\tf20_cfs
: call venvs\tf20_cfs\Scripts\activate
: pip install C:\\tmp\\tensorflow_pkg\\tensorflow-2.0.2-cp36-cp36m-win_amd64.whl
: pip install opencv-python
: pip install py-cpuinfo
: pip install pandas
: pip install WMI

: TF 2.1
virtualenv venvs\tf21
call venvs\tf21\Scripts\activate
pip install tensorflow==2.1.1
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 2.1
: virtualenv venvs\tf21_cfs
: call venvs\tf21_cfs\Scripts\activate
: pip install C:\\tmp\\tensorflow_pkg\\tensorflow-2.1.1-cp36-cp36m-win_amd64.whl
: pip install opencv-python
: pip install py-cpuinfo
: pip install pandas
: pip install WMI

: TF 2.1
virtualenv venvs\tf21_gpu
call venvs\tf21_gpu\Scripts\activate
pip install tensorflow-gpu==2.1.1
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 2.3
virtualenv venvs\tf23
call venvs\tf23\Scripts\activate
pip install tensorflow==2.3
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI

: TF 2.3
: virtualenv venvs\tf23_cfs
: call venvs\tf23_cfs\Scripts\activate
: pip install C:\\tmp\\tensorflow_pkg\\tensorflow-2.3.0-cp36-cp36m-win_amd64.whl
: pip install opencv-python
: pip install py-cpuinfo
: pip install pandas
: pip install WMI

: TF 2.3
virtualenv venvs\tf23_gpu
call venvs\tf23_gpu\Scripts\activate
pip install tensorflow-gpu==2.3
pip install opencv-python
pip install py-cpuinfo
pip install pandas
pip install WMI