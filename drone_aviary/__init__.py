import gym

gym.register(
    id="TakeoffAviary-v0",
    entry_point="drone_aviary.envs:TakeoffAviary",
)

gym.register(
    id="LandingAviary-v0",
    entry_point="drone_aviary.envs:LandingAviary",
)
