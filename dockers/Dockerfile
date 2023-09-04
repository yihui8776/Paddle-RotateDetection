# Version: 2.0.0

FROM registry.baidubce.com/paddlepaddle/paddle:2.4.1-gpu-cuda11.2-cudnn8.2-trt8.0

RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple


RUN pip3.7 install paddlehub --upgrade --no-cache-dir -i https://mirror.baidu.com/pypi/simple


RUN cd / && git clone https://gitee.com/paddlepaddle/PaddleDetection.git && cd /PaddleDetection  && \
         pip3.7 install -r requirements.txt --no-cache-dir -i https://mirror.baidu.com/pypi/simple && python3.7  ppdet/ext_op/setup.py install
#jupyter
RUN pip3.7 install --upgrade --no-cache-dir jinja2==3.0.3 protobuf==3.20.0  jupyter ipykernel==5.5.6   -i https://mirror.baidu.com/pypi/simple && ln -s /usr/local/python37/bin/jupyter /usr/bin/jupyter
RUN pip3.7 install --upgrade --no-cache-dir jupyter_contrib_nbextensions -i https://mirror.baidu.com/pypi/simple  && jupyter contrib nbextension install --user
#EXPOSE 8866
COPY jupyter_notebook_config.py /root/.jupyter/

COPY run_jupyter.sh /
RUN chmod +x  /run_jupyter.sh

RUN cd / && git clone https://gitee.com/paddlepaddle/PaddleOCR.git /PaddleOCR && cd /PaddleOCR && \
     pip3.7 install --no-cache-dir -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# IPython
EXPOSE 8888
# SSH
EXPOSE 22

CMD ["/run_jupyter.sh", "--allow-root"]

#RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v2.0_det_infer to ch_ppocr_server_v2.0_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py ）

#EXPOSE 8866

#CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]
