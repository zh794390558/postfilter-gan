FROM tensorflow/tensorflow:latest 

COPY sources.list /etc/apt
RUN groupadd -f -g 800  zhanghui && \
    groupadd -f -g 1022 atlas && \
    groupadd -f -g 1023 atlastts && \
    useradd -m -u 800 -g 800 -G 800,1022,1023 -d /home/zhanghui -m -s /bin/bash zhanghui &&\
    chown -R zhanghui:zhanghui /home/zhanghui && \
    apt-get update && \
    apt-get install -y  --no-install-recommends \
		vim git tig && \
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

USER zhanghui
WORKDIR  /home/zhanghui


