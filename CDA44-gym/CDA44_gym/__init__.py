from gym.envs.registration import register

register(
    id='CDA44_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='CDA44_gym.envs:CDA44Env',              # Expalined in envs/__init__.py
)
register(
    id='CDA44_env_extend-v0',
    entry_point='CDA44_gym.envs:ShishiEnvExtend',
)