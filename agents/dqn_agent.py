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
import sys
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--sparse', help='using sparse reward', type=str2bool, default=False)
parser.add_argument('--tnr', help='time negative reward', type=str2bool, default=False)
parser.add_argument('--mnr', help='move negative reward', type=str2bool, default=False)
parser.add_argument('-mr', '--max_rotating', help='using max rotation', type=str2bool, default=False)
parser.add_argument('-r', '--rl_rotating', help='using rotating in training', type=str2bool, default=True)
parser.add_argument('-t', '--test', help='just test', type=str2bool, default=False)
parser.add_argument('-tr', '--test_rotating', help='test_rotating', type=str2bool, default=True)
parser.add_argument('-uh', '--use_her', help='using HER', type=str2bool, default=False)
parser.add_argument('-ht', '--her_type', help='HER Type', type=str, default='future')
parser.add_argument('-hn', '--her_number', help='HER Number', type=int, default=4)
parser.add_argument('-n', '--name', help='Run Name', type=str, default='test_'+str(time.time()))
parser.add_argument('-m', '--map', help='Map Path', type=str, default='/home/nader/workspace/rl/gym-pathfinder/agents/maps/vertical_map/')
args = parser.parse_args()


env = gym.make('gym_pathfinder:PathFinder-v0', map_path=args.map)
env.sparse_reward = args.sparse
env.time_neg_reward = args.tnr
env.move_neg_reward = args.mnr
rl = DeepQ(train_interval_step=1, train_step_counter=32)
rl.create_model_cnn_dense()
rl.rotating = args.rl_rotating
rl.max_rotating = args.max_rotating
just_test = args.test
test_rot = args.test_rotating
test_dia = True
use_her = args.use_her
her_type = args.her_type
her_number = args.her_number
run_name = args.name

if not os.path.exists(run_name):
    os.makedirs(run_name)
if just_test:
    rl.read_weight(os.path.join('/home/nader/workspace/rl/gym-pathfinder/agents/results/jul24/testF', 'ep175.h5'))

f = open(os.path.join(run_name, 'model.txt'), 'w')
f.write(str(rl.model.to_json()))
f.close()
f = open(os.path.join(run_name, 'config.txt'), 'w')
f.write(str(args))
f.close()

train_episode = 1000
train_epoch = 200
test_episode = 100


def run_episode(is_test, ep, e, test_rot, test_dia):
    R = 0
    success = 0
    observations = []
    next_observations = []
    actions = []
    dones = []
    obs = env.reset_with_rot(test_rot, test_dia)
    obs = obs.reshape((10, 10, 1))
    if just_test:
        env.render()
    done = False
    while not done:
        action = rl.get_random_action(obs, 0.0 if is_test else 0.1)
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
        if use_her:
            for t in range(len(observations)):
                if dones[t]:
                    continue
                goals = []
                for s in range(t+1 if her_type == 'future' else 0, len(observations)):
                    goal = np.where(observations[s] == env.agent_value)
                    goals.append(goal)
                random.shuffle(goals)
                goals = goals[:her_number]
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
        for t in range(20):
            rl.train()
    if is_test:
        print(f'Epoch:{ep} Test Episode:{e} R:{R}')
    else:
        print(f'Epoch:{ep} Train Episode:{e} R:{R}')
    return R, success


def run_bunch(is_test, count_of_episodes, bunch_number, test_rot, test_dia):
    bunch_reward = 0
    bunch_success = 0
    for e in range(count_of_episodes):
        reward, success = run_episode(is_test, bunch_number, e, test_rot, test_dia)
        bunch_reward += reward
        bunch_success += success
    return bunch_reward, bunch_success


def process_results(rewards, successes, rot_rewards, rot_successes, dia_rewards, dia_successes):
    plt.plot(successes)
    plt.show()
    file = open(os.path.join(run_name, 'test_rewards.txt'), 'w')
    for x in rewards:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'test_successes.txt'), 'w')
    for x in successes:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'rot_test_rewards.txt'), 'w')
    for x in rot_rewards:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'rot_test_successes.txt'), 'w')
    for x in rot_successes:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'dia_test_rewards.txt'), 'w')
    for x in dia_rewards:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'dia_test_successes.txt'), 'w')
    for x in dia_successes:
        file.write(str(x) + '\n')
    file.close()
    file = open(os.path.join(run_name, 'loss.txt'), 'w')
    for x in rl.loss_values:
        file.write(str(x) + '\n')
    file.close()


def main():
    test_rewards = []
    test_success = []
    rot_test_rewards = []
    rot_test_success = []
    dia_test_rewards = []
    dia_test_success = []
    bunch_reward, bunch_success = run_bunch(True, test_episode, 0, False, False)
    test_rewards.append(bunch_reward / test_episode)
    test_success.append(bunch_success)
    if test_rot:
        bunch_reward, bunch_success = run_bunch(True, test_episode, 0, True, False)
        rot_test_rewards.append(bunch_reward / test_episode)
        rot_test_success.append(bunch_success)
    if test_dia:
        bunch_reward, bunch_success = run_bunch(True, test_episode, 0, False, True)
        dia_test_rewards.append(bunch_reward / test_episode)
        dia_test_success.append(bunch_success)
    rl.model.save_weights(os.path.join(run_name, f'ep{0}.h5'))

    for bunch_number in range(1, train_epoch + 1):
        if not just_test:
            run_bunch(False, train_episode, bunch_number, False)
            run_bunch(False, train_episode, bunch_number, False, False)
        bunch_reward, bunch_success = run_bunch(True, test_episode, bunch_number, False, False)
        test_rewards.append(bunch_reward / test_episode)
        test_success.append(bunch_success)
        if test_rot:
            bunch_reward, bunch_success = run_bunch(True, test_episode, bunch_number, True, False)
            rot_test_rewards.append(bunch_reward / test_episode)
            rot_test_success.append(bunch_success)
        if test_rot:
            bunch_reward, bunch_success = run_bunch(True, test_episode, bunch_number, False, True)
            dia_test_rewards.append(bunch_reward / test_episode)
            dia_test_success.append(bunch_success)
        rl.model.save_weights(os.path.join(run_name, f'ep{bunch_number}.h5'))
        if bunch_number % 10 == 0:
            process_results(test_rewards, test_success, rot_test_rewards, rot_test_success, dia_test_rewards,
                            dia_test_success)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        f = open(os.path.join(run_name, 'error.txt'), 'w')
        f.write(str(e))
        f.close()
