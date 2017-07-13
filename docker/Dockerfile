FROM tensorflow/tensorflow:latest 

RUN groupadd -g 800 -f zhanghui && \
    useradd -m -u 800 -g 800 -s /bin/bash zhanghui &&\
    mkdir -p /home/zhanghui &&\
    chown -R zhanghui:zhanghui /home/zhanghui

USER zhanghui
WORKDIR  /home/zhanghui


