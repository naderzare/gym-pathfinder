# f = open('/home/nader/workspace/rl/gym-pathfinder/agents/results/jul27/testH/loss.txt', 'r')
# loss = []
# for l in f:
#     loss.append(float(l))
# print(loss)
#
# import matplotlib.pyplot as plt
#
# plt.plot(loss[:])
# plt.show()
#

import os
import matplotlib.pyplot as plt
path = [['paper2/A1', 'paper2/A3'], ['paper2/B1', 'paper2/B2'],
        ['paper2/C1', 'paper2/C2'], ['paper2/D1', 'paper2/D2']]
colors = ['firebrick', 'darkorange', 'dodgerblue', 'limegreen']
avg_number = 1
for lp, c in zip(path, colors):
    acs = []
    for p in lp:
        file = os.path.join('/home/nader/workspace/rl/gym-pathfinder/agents/results/', p, 'dia_test_successes.txt')
        file = open(file, 'r').readlines()
        ac = [float(x) for x in file]
        av = []
        for i in range(int(200/avg_number)):
            av.append(sum(ac[i*avg_number:(i+1)*avg_number])/avg_number)
        acs.append(av)
    ac_max = [max(i, j) for i, j in zip(acs[0], acs[1])]
    ac_min = [min(i, j) for i, j in zip(acs[0], acs[1])]
    ac_avg = [(min(i, j) + max(i, j)) / 2.0 for i, j in zip(acs[0], acs[1])]
    plt.fill_between(range(len(acs[0])),ac_min, ac_max, alpha=0.5, color=c)
    plt.plot(ac_max, color=c)
    # plt.plot(acs[0], color=c)
    # plt.plot(acs[1], color=c)
plt.legend(['DQN', 'DQN_HER', 'DQN_Rot', 'DQN_HER_Rot'])
# plt.title('test in diagonal maps')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.ylim([0,100])
plt.xticks(range(0, int(200 / avg_number) + 1, avg_number * 25), range(0, 201, 25))
plt.show()