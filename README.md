conda create -n drones python=3.8
conda activate drones
pip3 install --upgrade pip
pip3 install setuptools==57.1.0
pip3 install -e .
pip3 install -r requirements.txt

