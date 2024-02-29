# 基础镜像
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu

# 避免在构建过程中出现任何交互式前端提示
ARG DEBIAN_FRONTEND=noninteractive

# 更新软件包列表并安装libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
