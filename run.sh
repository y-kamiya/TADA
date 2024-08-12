#!/bin/bash -eu

source $1/bin/activate
python -m apps.run --config configs/dev.yaml --text 'dummy'
