python -m venv --without-pip .venv-py311-docker
. .venv-py311-docker/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install --upgrade pip
./setup.sh .venv-py311-docker
