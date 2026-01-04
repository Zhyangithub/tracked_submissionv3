FROM --platform=linux/amd64 pytorch/pytorch

# 确保 Python 输出不被缓存
ENV PYTHONUNBUFFERED=1

# 创建非 root 用户 (Grand Challenge 推荐)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# --- 修改点：直接在这里安装依赖，防止 requirements.txt 漏写 ---
# scipy: 用于 model.py 的图像缩放和连通域分析
# simpleitk: 用于 inference.py 读取 .mha 文件
RUN python -m pip install \
    --user \
    --no-cache-dir \
    numpy \
    scipy \
    simpleitk

# 复制资源文件夹 (确保 best_model.pth 在里面)
COPY --chown=user:user resources /opt/app/resources

# 复制核心代码
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/

ENTRYPOINT ["python", "inference.py"]