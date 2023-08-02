from gymnasium.envs.registration import register

register(
    id='landing-aviary-v0',
    entry_point='drone_landing.envs:LandingAviary',
)
register(
    id='alignment-aviary-v0',
    entry_point='drone_landing.envs:AlignmentAviary',
)

