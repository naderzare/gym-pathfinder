import gym

env = gym.make('gym_pathfinder:PathFinder-v0')

obs = env.reset()
env.render()
print(obs)
done = False
while not done:
    action = int(input('enter (0 Down, 1 Left, 2 Up, 3 Right:'))
    actions = [0, 0, 0, 0, 1, 0, 3, 0, 2]
    action = actions[action]
    obs, reward, done, info = env.step(action)
    env.render()
    print(reward)

env.close()