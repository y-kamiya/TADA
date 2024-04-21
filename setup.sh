#!/bin/bash -ex

pip install --upgrade pip  
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install wheel
pip install -r requirements.txt -r ImageDream/requirements.txt --no-build-isolation
 
pushd smplx
python setup.py install
popd

pip install -e ImageDream
pip install -e ImageDream/extern/ImageDream

# can not download due to permission
# pip install gdown 
# mkdir -p data/omnidata 
# pushd data/omnidata 
# gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
# gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
# popd
