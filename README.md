# AINUC-gettingStart
AINUC说明





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


refernce:

https://www.cnblogs.com/ag-chen/articles/13577020.html
