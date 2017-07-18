FROM tensorflow/tensorflow:latest 

COPY sources.list /etc/apt
RUN groupadd -f -g 800  zhanghui && \
    groupadd -f -g 1022 atlas && \
    groupadd -f -g 1023 atlastts && \
    useradd -m -u 800 -g 800 -G 800,1022,1023 -d /home/zhanghui -m -s /bin/bash zhanghui &&\
    chown zhanghui:zhanghui -R /home/zhanghui && \
    echo 'zhanghui:zhanghui' | chpasswd && \
    apt-get update && \
    apt-get install -y  --no-install-recommends --allow-remove-essential \
	vim git bash-completion tig tree ctags sudo && \
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* && \
    usermod -a -G sudo zhanghui && \
    git clone --depth=1 https://github.com/amix/vimrc.git /home/zhanghui/.vim_runtime && \
    chown zhanghui:zhanghui -R /home/zhanghui/.vim_runtime && \
    sed -i '$a sh ~/.vim_runtime/install_awesome_vimrc.sh' /home/zhanghui/.profile
    

USER zhanghui
WORKDIR  /home/zhanghui

