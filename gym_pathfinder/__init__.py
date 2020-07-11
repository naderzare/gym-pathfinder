from gym.envs.registration import register

register(
    id='PathFinder-v0',
    entry_point='gym_pathfinder.envs:PathFinderEnv',
)
