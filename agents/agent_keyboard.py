import gym

env = gym.make('gym_pathfinder:pathfinder-v0')
env.add_map_path('horizontal', '/home/nader/workspace/rl/gym-pathfinder/agents/maps/vertical_map/', 'vertical')
env.add_map_path('diagonal', '/home/nader/workspace/rl/gym-pathfinder/agents/maps/diagonal_map/')
obs = env.reset(map_name='vertical')
env.sparse_reward = False
env.abs_normal_reward = 0.02
env.move_neg_reward = True
env.render()
done = False
while not done:
    action = int(input('enter (0 Down, 1 Left, 2 Up, 3 Right:'))
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render()

env.close()