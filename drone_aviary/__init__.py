from gymnasium.envs.registration import register

register(
    id='takeoff-aviary-v0',
    entry_point='drone_aviary.envs:TakeoffAviary',
)

register(
    id='landing-aviary-v0',
    entry_point='drone_aviary.envs:LandingAviary',
)


