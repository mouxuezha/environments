from gym.envs.registration import register

register(
    id='CDA542_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='CDA542_gym.envs:CDA542Env',              # Expalined in envs/__init__.py
)
register(
    id='CDA542_env_extend-v0',
    entry_point='CDA542_gym.envs:ShishiEnvExtend',
)