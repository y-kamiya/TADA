FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt update -y && apt install -y git libgl1-mesa-dev libglib2.0-0
