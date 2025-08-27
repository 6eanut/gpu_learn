#!/bin/bash

# 脚本用于在cuda learning里面用ncu

# 安装系统依赖包
apt update
apt install -y python3-dev curl libssl-dev libbz2-dev libreadline-dev \
libsqlite3-dev liblzma-dev libffi-dev tk-dev libncurses5-dev libncursesw5-dev

# 安装pyenv
curl -fsSL https://pyenv.run | bash

# 配置环境变量
cat >> ~/.bashrc << 'EOF'
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
EOF

# 应用环境变量
source ~/.bashrc

# 安装指定版本的Python
pyenv install 3.9.6

# 设置全局Python版本
pyenv global 3.9.6

# 安装PyTorch及相关库
pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio==0.10.2+cu102 \
-f https://download.pytorch.org/whl/cu102/torch_stable.html

# 安装其他Python包
pip install numpy==1.10.2 pandas triton matplotlib