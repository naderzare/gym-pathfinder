from gym.envs.registration import register

register(
    id='ImagePath-v0',
    entry_point='gym_imagepath.envs:ImagePathEnv',
)
