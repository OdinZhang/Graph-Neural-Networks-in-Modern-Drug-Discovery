from setuptools import find_packages, setup

# ============================================================
# TxGNN 包安装配置
#
# 课堂安装说明（按顺序执行）：
#
# 1. 创建环境：
#    conda create --name txgnn_env python=3.8
#    conda activate txgnn_env
#
# 2. 安装 PyTorch（根据 GPU 版本选择）：
#    https://pytorch.org/get-started/locally/
#
# 3. 安装 DGL（必须使用 0.5.2 版本）：
#    conda install -c dglteam dgl-cuda<版本>==0.5.2
#    # 无 GPU：conda install -c dglteam dgl==0.5.2
#
# 4. 从本地源码安装（开发模式，修改源码立即生效）：
#    pip install -e .
#
# ============================================================

from os import path
from io import open

# 读取版本号
ver_file = path.join('txgnn', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()


with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='TxGNN',
    version=__version__,
    license='MIT',
    description='TxGNN: Zero-shot drug repurposing with geometric deep learning on biomedical knowledge graphs',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/mims-harvard/TxGNN',
    author='TxGNN Team',
    author_email='kexinh@stanford.edu',
    packages=find_packages(exclude=['test', 'reproduce']),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    python_requires='>=3.8',
)