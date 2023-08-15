conda create -n drones python=3.8
conda activate drones
pip3 install --upgrade pip
pip3 install setuptools==57.1.0
pip3 install -e .
pip3 install -r requirements.txt

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html