from gym.envs.registration import register

register(
    id='NACA652_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='NACA652_gym.envs:NACA652Env',              # Expalined in envs/__init__.py
)
register(
    id='NACA652_env_extend-v0',
    entry_point='NACA652_gym.envs:ShishiEnvExtend',
)