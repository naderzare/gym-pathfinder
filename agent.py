import gym

env = gym.make('gym_pathfinder:PathFinder-v0')

obs = env.reset()
env.render()
print(obs)
done = False
while not done:
    action = int(input('enter (0 Down, 1 Left, 2 Up, 3 Right:'))
    obs, reward, done = env.step(action)
    env.render()
    print(reward)

env.close()