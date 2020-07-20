import gym

env = gym.make('gym_pathfinder:PathFinder-v0', map_path='/home/nader/workspace/rl/gym-pathfinder/agents/maps/vertical_map/')

obs = env.reset()
env.render()
done = False
while not done:
    action = int(input('enter (0 Down, 1 Left, 2 Up, 3 Right:'))
    actions = [0, 0, 0, 0, 1, 0, 3, 0, 2]
    action = actions[action]
    obs, reward, done, info = env.step(action)
    env.render()

env.close()