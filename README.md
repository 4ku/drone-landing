# Drone training repository

## Setup

### Virtual environment
#### Prerequisites
* Python 3.8
* CUDA 11.8
#### Setup
```
conda create -n drones python=3.8
conda activate drones
pip3 install --upgrade pip
pip3 install setuptools==57.1.0
pip3 install -e .
pip3 install -r requirements.txt

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
```

### Docker
docker run --gpus ...

## Environments
Special aviary for drone training was created: big cube 10 by 10 by 10 meters with a platform with aruco marker in a center. An image of that aviary you can see here:
![](imgs/Aviary.png)

Based on this aviary 2 environments were created: 
1) `TakeoffAviary` - drone starts from the platform and have to go as high as possible for some period of time.
---------------- Гифки с идеальным полиси со стороны
2) `LandingAviary` - drone have to land smoothly to the platform with aruco marker.
---------------- Гифки с идеальным полиси со стороны

## Methods
3 different approaches were tested: PPO from stable_baselines3, model-based approach - dreamerV3 and imitation learning.


## Results

### PPO
#### TakeoffAviary
![](imgs/takeoff_PPO_results.png)

#### LandingAviary
![](imgs/landing_PPO_results.png)

### DreamerV3
#### TakeoffAviary

#### LandingAviary
No reason to make tests, because TakeoffAviary don't work.

### Imitation learning
#### TakeoffAviary

#### LandingAviary