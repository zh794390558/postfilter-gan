FROM tensorflow/tensorflow:latest 

RUN groupadd -f -g 800  zhanghui && \
    groupadd -f -g 1022 atlas && \
    groupadd -f -g 1023 atlastts && \
    useradd -m -u 800 -g 800 -G 1022,1023 -d /home/zhanghui -m -s /bin/bash zhanghui &&\
    chown -R zhanghui:zhanghui /home/zhanghui

USER zhanghui
WORKDIR  /home/zhanghui


