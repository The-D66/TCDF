FROM docker.the-d.fun/pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /app

# 配置apt使用清华TUNA镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
  sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装常用中文字体
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  fonts-noto-cjk \
  fonts-noto-cjk-extra \
  xfonts-intl-chinese \
  xfonts-wqy \
  ttf-wqy-microhei \
  ttf-wqy-zenhei && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 配置pip使用清华TUNA镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装依赖，但不重复安装PyTorch（基础镜像已包含）
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt && \
  pip install --no-cache-dir -r requirements_no_torch.txt

# 为Jupyter设置
EXPOSE 8888

# 设置中文支持
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# 创建非root用户以提高安全性
RUN useradd -m jupyter
USER jupyter

# 设置工作目录权限
RUN mkdir -p /home/jupyter/work
WORKDIR /home/jupyter/work

# 启动Jupyter
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"] 