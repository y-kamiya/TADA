#!/bin/bash -ex

pip install --upgrade pip  
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r ImageDream/requirements.txt
 
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
