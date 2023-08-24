import gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space):
        super(CustomCNN, self).__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # If input (in case if n_stack=3): `torch.Size([1, 12, 64, 64])` then 
        # After Conv1:  torch.Size([1, 32, 15, 15])
        # After Conv2:  torch.Size([1, 64, 6, 6])
        # After Conv3:  torch.Size([1, 128, 4, 4])
        # After Conv4:  torch.Size([1, 64, 4, 4])
        # After flatten:  torch.Size([1, 1024])

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.features_dim = n_flatten

    def forward(self, observations: torch.Tensor):
        return self.cnn(observations)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args, **kwargs, net_arch=[128, 128], features_extractor_class=CustomCNN
        )
