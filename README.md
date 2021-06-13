# AINUC-gettingStart
AINUC说明

## 所提供的资料在启动后桌面上有说明

## Python测试运行
```bash
pip install openvino
```
```py
from openvino.inference_engine import IENetwork
```
如果出现
```bash
ImportError: DLL load failed:
```
原因是找不到dll文件, 可以把openvino的bin路径加入环境变量
``` bash
C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release
C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Debug
```
还是没有用的话, 可能就是没有初始化 [power shell]
```bash
cd 'C:\Program Files (x86)\IntelSWTools\openvino\bin\'
.\setupvars.bat
```


## 案例：
https://github.com/radiumray/AINUC-gettingStart/blob/main/faceDeteClass.py


## 模型转换：

+ /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites$ sudo ./install_prerequisites.sh

+ onnx---->xml
### python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py --input_model singleHumanPose.onnx --input_shape [1,3,384,288] --input "input" --output "output"


+ caffe----->xml
### python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py  --input_model MobileNetSSD_deploy.caffemodel --scale 127.5


+ pb----->xml
### python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py  --input_model mask_recognize.h5.pb --input_shape=[1,160,160,3]




+ 将PyTorch模型转换为ONNX格式，使用下载的仓库中的convert_to_onnx.py脚本，进行模型格式转换：
### python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT>. It produces human-pose-estimation.onnx



## refernce:

+ https://www.cnblogs.com/ag-chen/articles/13577020.html
+ https://software.intel.com/content/www/cn/zh/develop/tools/openvino-toolkit.html
