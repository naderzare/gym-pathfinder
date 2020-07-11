import gym

env = gym.make('gym_pathfinder:PathFinder-v0')

obs = env.reset()
env.render()
print(obs)
done = False
while not done:
    action = int(input('enter:'))
    obs, reward, done = env.step(action)
    env.render()
    print(reward)

env.close()