#!/bin/bash -e
#
conda config --add pkgs_dirs ../../envs/tada/pkgs
conda env create --file environment.yml --prefix ../../envs/tada/conda
conda activate tada 
pip install -r requirements.txt
 
pushd smplx
python setup.py install
popd

conda install -c conda-forge pytorch-lightening
pip install cuda-pytorch

mkdir -p data/omnidata 
pushd data/omnidata 
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
popd
