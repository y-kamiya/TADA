#!/bin/bash -ex

pip install --upgrade pip 
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
 
pushd smplx
python setup.py install
popd

pip install -r ImageDream/requirements.txt
pip install -e ImageDream
pip install -e ImageDream/extern/ImageDream
# pip install cuda-python pytorch-lightning omegaconf

# can not download due to permission
# pip install gdown 
# mkdir -p data/omnidata 
# pushd data/omnidata 
# gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
# gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
# popd
