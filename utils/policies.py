import gymnasium as gym
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
        # If input: `torch.Size([1, 12, 48, 64])` then
        # After Conv1:  torch.Size([1, 32, 11, 15])
        # After Conv2:  torch.Size([1, 64, 4, 6])
        # After Conv3:  torch.Size([1, 128, 2, 4])
        # After Conv4:  torch.Size([1, 64, 2, 4])
        # After flatten:  torch.Size([1, 512])

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.features_dim = n_flatten

    def forward(self, observations: torch.Tensor):
        return self.cnn(observations)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[128, 128],  
                                           features_extractor_class=CustomCNN)



# import gymnasium as gym
# import torch 
# import torch.nn as nn
# from stable_baselines3.common.policies import ActorCriticPolicy

# class CustomCNN(nn.Module):
#     def __init__(self, observation_space: gym.spaces.Space):
#         super(CustomCNN, self).__init__()
#         n_input_channels = observation_space.shape[0]  # get the number of input channels from observation space
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())

#         # Set the dimensionality of the output features
#         self.features_dim = 64  # <-- This value needs to be the output dimension of your feature extractor

#     def forward(self, observations: torch.Tensor):
#         return self.linear(self.cnn(observations))


# class CustomPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs,
#                                            net_arch=[64, 64],  # MLP after the features extractor
#                                            features_extractor_class=CustomCNN)