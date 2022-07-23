from gym.envs.registration import register

register(
    id='CDA43_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='CDA43_gym.envs:CDA43Env',              # Expalined in envs/__init__.py
)
register(
    id='CDA43_env_extend-v0',
    entry_point='CDA43_gym.envs:ShishiEnvExtend',
)