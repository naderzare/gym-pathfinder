import gym
from gym_pathfinder.envs.pathfinder_env import PathFinderEnv
env = PathFinderEnv(20, 20)
env.add_map_path('horizontal', '/home/nader/workspace/rl/gym-pathfinder/agents/maps/20_20/vertical_map/', 'vertical')
env.add_map_path('diagonal', '/home/nader/workspace/rl/gym-pathfinder/agents/maps/20_20/diagonal_map/')
obs = env.reset(map_name='horizontal')
env.sparse_reward = False
env.abs_normal_reward = 0.02
env.move_neg_reward = False
env.render()
done = False
while not done:
    action = int(input('enter (0 Down, 1 Left, 2 Up, 3 Right:'))
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render()

env.close()