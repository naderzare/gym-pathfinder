import gym
from agents.dqn import DeepQ
import copy
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import random
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
parser.add_argument('-maxr', help='max reward', type=float, default=2)
parser.add_argument('-minr', help='min reward', type=float, default=-2)
parser.add_argument('-normr', help='normal reward', type=float, default=0.02)
parser.add_argument('-tnr', help='time negative reward', type=str2bool, default=False)
parser.add_argument('-mnr', help='move negative reward', type=str2bool, default=False)
parser.add_argument('-mr', help='using max rotation', type=str2bool, default=False)
parser.add_argument('-mrf', help='Rotation function', type=str, default='max')
parser.add_argument('-rl', help='RL type', type=str, default='dqn')
parser.add_argument('-r', help='using rotating in training', type=str2bool, default=True)
parser.add_argument('-t', '--test', help='just test', type=str2bool, default=False)
parser.add_argument('-tr', '--test_rotating', help='test_rotating', type=str2bool, default=True)
parser.add_argument('-uh', '--use_her', help='using HER', type=str2bool, default=False)
parser.add_argument('-ht', '--her_type', help='HER Type', type=str, default='future')
parser.add_argument('-hn', '--her_number', help='HER Number', type=int, default=4)
parser.add_argument('-n', '--name', help='Run Name', type=str, default='test_'+str(time.time()))
parser.add_argument('-si', help='size i', type=int, default=10)
parser.add_argument('-sj', help='size j', type=int, default=10)
args = parser.parse_args()

env = gym.make('gym_pathfinder:pathfinder-v0', count_i=args.si, count_j=args.sj)
# env = PathFinderEnv(args.si, args.sj)

env.add_map_path('horizontal', './maps/20_20/vertical_map/', 'vertical')
env.add_map_path('diagonal', './maps/20_20/diagonal_map/')

env.sparse_reward = args.sparse
env.time_neg_reward = args.tnr
env.min_reward = args.minr
env.max_reward = args.maxr
env.episode_max_cycle = 100

env.abs_normal_reward = args.normr
rl = DeepQ(train_interval_step=1, train_step_counter=32)
rl.create_model_cnn_dense()
rl.rotating = args.r
rl.max_rotating = args.mr
rl.max_rotating_function = args.mrf
rl.rl_type = args.rl
rl.input_shape_i = args.si
rl.input_shape_j = args.sj
just_test = args.test
test_rot = args.test_rotating
test_dia = True
use_her = args.use_her
her_type = args.her_type
her_number = args.her_number
run_name = args.name

if args.r:
    run_name += '_rot'
else:
    run_name += '_simple'
if args.mr:
    run_name += '_'
    run_name += args.mrf
else:
    run_name += '_not'
if args.sparse:
    run_name += '_sparse'
else:
    run_name += '_contin'
if args.mnr:
    run_name += '_neg'
else:
    run_name += '_pos'
run_name += '_r' + str(args.minr) + '_' + str(args.maxr) + '_' + str(args.normr) + '_' + args.rl
print(run_name)
if not os.path.exists(run_name):
    os.makedirs(run_name)
if just_test:
    rl.read_weight(os.path.join('/home/nader/workspace/rl/gym-pathfinder/agents/results/jul24/testF', 'ep175.h5'))

f = open(os.path.join(run_name, 'model.txt'), 'w')
f.write(str(rl.online_network.to_json()))
f.close()
f = open(os.path.join(run_name, 'config.txt'), 'w')
f.write(str(args))
f.close()

train_episode = 1000
train_epoch = 100
test_episode = 100


def run_episode(is_test, ep, e, map_name):
    dif_q_r = 0
    R = 0
    rewards = []
    q_values = []
    success = 0
    observations = []
    next_observations = []
    actions = []
    dones = []
    obs = env.reset(map_name)
    obs = obs.reshape((args.si, args.sj, 1))
    if just_test:
        env.render()
    done = False
    while not done:
        action, q_value = rl.get_random_action(obs, 0.0 if is_test else 0.1)
        prev_obs = copy.copy(obs)
        obs, reward, done, info = env.step(action)
        if type(q_value) is int:
            q_values.append(q_value)
        else:
            q_values.append(q_value[0])
        rewards.append(reward)
        R += reward
        if info['result'] == 'goal':
            success = 1
        obs = obs.reshape((args.si, args.sj, 1))
        if not is_test:
            if info['result'] == 'time':
                rl.add_to_buffer(prev_obs, action, reward, obs, False, False, True)
                dones.append(False)
            else:
                rl.add_to_buffer(prev_obs, action, reward, obs, done, False, done)
                dones.append(done)
            observations.append(prev_obs)
            next_observations.append(obs)
            actions.append(action)
        if just_test:
            env.render()
            time.sleep(0.2)
    if is_test:
        all_rewards = sum([rewards[i] * (i + 1) for i in range(len(rewards))])
        all_q_values = sum(q_values)
        dif_q_r = all_q_values - all_rewards
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
        for t in range(40):
            rl.train()
    if is_test:
        print(f'Epoch:{ep} Test Episode:{e} R:{R}')
    else:
        print(f'Epoch:{ep} Train Episode:{e} R:{R}')
    return R, success, dif_q_r


def run_bunch(is_test, count_of_episodes, bunch_number, map_name):
    bunch_reward = 0
    bunch_success = 0
    bunch_dif_q_r = 0
    for e in range(count_of_episodes):
        reward, success, dif_q_r = run_episode(is_test, bunch_number, e, map_name)
        bunch_reward += reward
        bunch_success += success
        bunch_dif_q_r += dif_q_r
    return bunch_reward, bunch_success, bunch_dif_q_r


def process_results(rewards, successes, dif_q_r):
    keys = list(successes.keys())
    plt.plot(successes[keys[0]])
    # plt.show()
    for k in keys:
        file = open(os.path.join(run_name, k + '_difqr.txt'), 'w')
        for x in dif_q_r[k]:
            file.write(str(x) + '\n')
        file.close()
        file = open(os.path.join(run_name, k + '_rewards.txt'), 'w')
        for x in rewards[k]:
            file.write(str(x) + '\n')
        file.close()
        file = open(os.path.join(run_name, k + '_successes.txt'), 'w')
        for x in successes[k]:
            file.write(str(x) + '\n')
        file.close()
    file = open(os.path.join(run_name, 'loss.txt'), 'w')
    # for x in rl.loss_values:
    #     file.write(str(x) + '\n')
    # file.close()


def main():
    train_maps = ['horizontal']
    test_maps = ['horizontal', 'vertical', 'diagonal']
    test_results = {x: 0 for x in test_maps}
    test_rewards = {x: [] for x in test_maps}
    test_success = {x: [] for x in test_maps}
    test_dif_q_r = {x: [] for x in test_maps}
    for test_map in test_maps:
        bunch_reward, bunch_success, bunch_dif_q_r = run_bunch(True, test_episode, 0, test_map)
        test_rewards[test_map].append(bunch_reward / test_episode)
        test_success[test_map].append(bunch_success)
        test_dif_q_r[test_map].append(bunch_dif_q_r / test_episode)
    process_results(test_rewards, test_success, test_dif_q_r)

    for bunch_number in range(1, train_epoch + 1):
        if not just_test:
            for train_map in train_maps:
                run_bunch(False, train_episode, bunch_number, train_map)
        for test_map in test_maps:
            bunch_reward, bunch_success, bunch_dif_q_r = run_bunch(True, test_episode, bunch_number, test_map)
            test_rewards[test_map].append(bunch_reward / test_episode)
            test_success[test_map].append(bunch_success)
            test_dif_q_r[test_map].append(bunch_dif_q_r / test_episode)
            if bunch_success > test_results[test_map]:
                test_results[test_map] = bunch_success
                rl.online_network.save_weights(os.path.join(run_name, f'best_{test_map}.h5'))
        if bunch_number % 10 == 0:
            process_results(test_rewards, test_success, test_dif_q_r)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        f = open(os.path.join(run_name, 'error.txt'), 'w')
        f.write(str(e))
        f.close()
