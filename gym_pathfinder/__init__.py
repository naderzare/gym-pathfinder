from gym.envs.registration import register

register(
    id='pathfinder-v0',
    entry_point='gym_pathfinder.envs:PathFinderEnv',
)
