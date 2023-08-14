import gym

gym.register(
    id="takeoff-aviary-v0",
    entry_point="drone_aviary.envs:TakeoffAviary",
)

gym.register(
    id="landing-aviary-v0",
    entry_point="drone_aviary.envs:LandingAviary",
)
