FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
MAINTAINER hkw

COPY . /workspace/
COPY sources.list /etc/apt/sources.list

RUN apt-get update
RUN apt-get install -y python3-gdbm libglib2.0-dev
RUN apt-get install libgl1-mesa-glx -y
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
