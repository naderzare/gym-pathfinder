import gym
from agents.dqn import DeepQ
import copy
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.49
set_session(tf.Session(config=config))

env = gym.make('gym_pathfinder:PathFinder-v0', map_path='/home/nader/workspace/rl/gym-pathfinder/agents/maps/vertical_map/')
env.sparse_reward = False
rl = DeepQ(train_interval_step=1, train_step_counter=32)
rl.create_model_cnn_dense()
just_test = False
use_her = False
run_name = 'human_maps_verti99'

if not os.path.exists(run_name):
    os.makedirs(run_name)
if just_test:
    rl.read_weight(os.path.join(run_name, 'ep180.h5'))


train_episode = 1000
train_epoch = 200
test_episode = 100


def run_episode(is_test, ep, e):
    R = 0
    success = 0
    obs = env.reset()
    obs = obs.reshape((10, 10, 1))
    if just_test:
        env.render()
    done = False
    while not done:
        action = rl.get_random_action(obs, 0.1)
        prev_obs = copy.copy(obs)
        obs, reward, done, info = env.step(action)
        R += reward
        if info['result'] == 'goal':
            success = 1
        obs = obs.reshape((10, 10, 1))
        if not is_test:
            if info['result'] == 'time':
                rl.add_to_buffer(prev_obs, action, reward, obs, False)
            else:
                rl.add_to_buffer(prev_obs, action, reward, obs, done)
        if just_test:
            env.render()
            time.sleep(0.2)
    if is_test:
        print(f'Epoch:{ep} Test Episode:{e} R:{R}')
    else:
        print(f'Epoch:{ep} Train Episode:{e} R:{R}')
    return R, success


def run_episode_her(is_test, ep, e):
    R = 0
    success = 0
    observations = []
    next_observations = []
    actions = []
    dones = []
    obs = env.reset()
    obs = obs.reshape((10, 10, 1))
    if just_test:
        env.render()
    done = False
    while not done:
        action = rl.get_random_action(obs, 0.1)
        prev_obs = copy.copy(obs)
        obs, reward, done, info = env.step(action)
        R += reward
        if info['result'] == 'goal':
            success = 1
        obs = obs.reshape((10, 10, 1))
        if not is_test:
            if info['result'] == 'time':
                rl.add_to_buffer(prev_obs, action, reward, obs, False, False)
                dones.append(False)
            else:
                rl.add_to_buffer(prev_obs, action, reward, obs, done, False)
                dones.append(done)
            observations.append(prev_obs)
            next_observations.append(obs)
            actions.append(action)
        if just_test:
            env.render()
            time.sleep(0.2)
    if not is_test:
        for t in range(len(observations)):
            if dones[t]:
                continue
            goals = []
            for s in range(t+1, len(observations)):
                goal = np.where(observations[s] == env.agent_value)
                goals.append(goal)
            random.shuffle(goals)
            goals = goals[:]
            for g in goals:
                obs = np.copy(observations[t])
                next_obs = np.copy(next_observations[t])
                action = actions[t]
                goal = np.where(obs == env.goal_value)
                agent = np.where(obs == env.agent_value)
                if agent == g:
                    continue
                obs[goal] = env.free_value
                next_obs[goal] = env.free_value
                obs[g] = env.goal_value
                next_obs[g] = env.goal_value
                reward, done = env.reward_calculator(np.copy(obs), np.copy(next_obs))
                if type(reward) == np.ndarray:
                    reward = float(reward[0])
                rl.add_to_buffer(obs, action, reward, next_obs, done, False)
        for t in range(len(observations)):
            rl.train()
    if is_test:
        print(f'Epoch:{ep} Test Episode:{e} R:{R}')
    else:
        print(f'Epoch:{ep} Train Episode:{e} R:{R}')
    return R, success


def run_bunch(is_test, count_of_episodes, bunch_number):
    bunch_reward = 0
    bunch_success = 0
    for e in range(count_of_episodes):
        if use_her:
            reward, success = run_episode_her(is_test, bunch_number, e)
        else:
            reward, success = run_episode(is_test, bunch_number, e)
        bunch_reward += reward
        bunch_success += success
    return bunch_reward, bunch_success


def process_results(rewards, successes):
    plt.plot(successes)
    plt.show()
    file = open(os.path.join(run_name, 'test_rewards'), 'w')
    for x in rewards:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'test_successes'), 'w')
    for x in successes:
        file.write(str(x) + '\n')
    file.close()


test_rewards = []
test_success = []
bunch_reward, bunch_success = run_bunch(True, test_episode, 0)
rl.model.save_weights(os.path.join(run_name, f'ep{0}.h5'))
test_rewards.append(bunch_reward / test_episode)
for bunch_number in range(1, train_epoch + 1):
    if not just_test:
        run_bunch(False, train_episode, bunch_number)
    bunch_reward, bunch_success = run_bunch(True, test_episode, bunch_number)
    rl.model.save_weights(os.path.join(run_name, f'ep{bunch_number}.h5'))
    test_rewards.append(bunch_reward / test_episode)
    test_success.append(bunch_success)
    if bunch_number % 2 == 0:
        process_results(test_rewards, test_success)
