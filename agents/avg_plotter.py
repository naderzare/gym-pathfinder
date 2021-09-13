import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
from scipy import stats


font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 10}
plt.rcParams['axes.facecolor'] = 'whitesmoke'
matplotlib.rc('font', **font)
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

main_paths = ['results/big/big1', 'results/big/big2', 'results/big/big3']
# {'code': '', 'avg': [], 'max': [], 'min':[], 'names': []}
a_c = '[1-9a-zA-Z_]*'
codes = [
    'not_contin_pos',
    'not_contin_neg',
    'not_sparse_pos',
]
# codes = [
#     'simple_not_sparse_pos',
#     'simple_not_contin_pos',
#     'simple_not_contin_neg',
# ]
codes = [
    f'rot_not',
    f'rot_fir',
    f'rot_max',
    f'rot_avg',
    f'rot_min',
    f'rot_random',
    f'simple',
]
data = []
for c in codes:
    data.append({'code': c, 'avg': {'h': [], 'v': [], 'd': []}, 'max': {'h': [], 'v': [], 'd': []},
                 'min': {'h': [], 'v': [], 'd': []}, 'data': {'h': [], 'v': [], 'd': []}, 'names': []})

block_code = ['t42_', 't45_', 't46_', 't43_', 't44_', '-2.0_0.0_', 'neg[1-9a-zA-Z_]*-2.0_0.0', '-0.1_2.0_0.02', '0.0_2.0_0.02']
block_code = []
line_function = 'avg'

def read_data():
    for main_path in main_paths:
        if not os.path.exists(main_path):
            continue
        directories = os.listdir(main_path)
        for d in directories:
            is_block = False
            for bc in block_code:
                if re.search(bc, d):
                    is_block = True
            if is_block:
                continue
            path = os.path.join(main_path, d)
            match = -1
            for k in range(len(data)):
                if re.search(data[k]['code'], d):
                    print(main_path, d, data[k]['code'], re.search(data[k]['code'], d))
                    match = k
                    break
            if match == -1:
                continue
            f_h = open(os.path.join(path, 'horizontal_successes.txt')).readlines()
            f_v = open(os.path.join(path, 'vertical_successes.txt')).readlines()
            f_d = open(os.path.join(path, 'diagonal_successes.txt')).readlines()
            data[match]['data']['h'].append([])
            data[match]['data']['v'].append([])
            data[match]['data']['d'].append([])
            for l in range(len(f_h)):
                data[match]['data']['h'][-1].append(float(f_h[l]))
                data[match]['data']['v'][-1].append(float(f_v[l]))
                data[match]['data']['d'][-1].append(float(f_d[l]))
    for d in data:
        if len(d['data']['h']) == 0:
            continue
        for i in range(len(d['data']['h'][0])):
            d['avg']['h'].append(0)
            d['max']['h'].append(0)
            d['min']['h'].append(0)
            d['avg']['h'][i] = 0
            d['max']['h'][i] = 0
            d['min']['h'][i] = 100
            for j in range(len(d['data']['h'])):
                d['avg']['h'][i] += d['data']['h'][j][i]
                d['max']['h'][i] = max(d['max']['h'][i], d['data']['h'][j][i])
                d['min']['h'][i] = min(d['min']['h'][i], d['data']['h'][j][i])
            d['avg']['h'][i] /= len(d['data']['h'])
    for d in data:
        if len(d['data']['v']) == 0:
            continue
        for i in range(len(d['data']['v'][0])):
            d['avg']['v'].append(0)
            d['max']['v'].append(0)
            d['min']['v'].append(0)
            d['avg']['v'][i] = 0
            d['max']['v'][i] = 0
            d['min']['v'][i] = 100
            for j in range(len(d['data']['v'])):
                d['avg']['v'][i] += d['data']['v'][j][i]
                d['max']['v'][i] = max(d['max']['v'][i], d['data']['v'][j][i])
                d['min']['v'][i] = min(d['min']['v'][i], d['data']['v'][j][i])
            d['avg']['v'][i] /= len(d['data']['v'])
    for d in data:
        if len(d['data']['d']) == 0:
            continue
        for i in range(len(d['data']['d'][0])):
            d['avg']['d'].append(0)
            d['max']['d'].append(0)
            d['min']['d'].append(0)
            d['avg']['d'][i] = 0
            d['max']['d'][i] = 0
            d['min']['d'][i] = 100
            for j in range(len(d['data']['d'])):
                d['avg']['d'][i] += d['data']['d'][j][i]
                d['max']['d'][i] = max(d['max']['d'][i], d['data']['d'][j][i])
                d['min']['d'][i] = min(d['min']['d'][i], d['data']['d'][j][i])
            d['avg']['d'][i] /= len(d['data']['d'])

read_data()
# print(data[2]['data'])
colors = ['red', 'blue', 'green', 'orange', 'black', 'gray', 'pink']
c = 0
fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0})
fig.set_figheight(4)
fig.set_figwidth(15)
leg = []
for d in data:
    if len(d['avg']['h']) > 0:
        print(d['code'])
        print(d.keys())
        print(len(d[line_function]['h']))
        axes[0].plot(d[line_function]['h'], color=colors[c])
        axes[0].fill_between(range(0, len(d['avg']['h'])), d['min']['h'], d['max']['h'], color=colors[c], alpha=0.1)
        axes[0].set_title('H')
        axes[0].set_xlim([0, 100])

        axes[1].plot(d[line_function]['v'], color=colors[c])
        axes[1].fill_between(range(0, len(d['avg']['v'])), d['min']['v'], d['max']['v'], color=colors[c], alpha=0.1)
        axes[1].set_title('V')
        axes[1].set_xlim([0, 100])

        axes[2].plot(d[line_function]['d'], color=colors[c])
        axes[2].fill_between(range(0, len(d['avg']['d'])), d['min']['d'], d['max']['d'], color=colors[c], alpha=0.1)
        axes[2].set_title('D')
        axes[2].set_xlim([0, 100])
        leg.append(d['code'])
        c += 1
print(leg)
plt.legend(leg, bbox_to_anchor=(1.05, 1), loc='upper left')

# import matplotlib.patches as patches
# rs = []
# for l in range(len(leg)):
#     rs.append(patches.Rectangle((0,0),1,1,facecolor=colors[l]))
# plt.legend(rs, leg)

plt.show()

for a in data:
    for b in data:
        if a['code'] == b['code']:
            continue
        if len(a['avg']['h']) == 0:
            continue
        t, p = stats.ttest_ind(a['avg']['h'][-50:], b['avg']['h'][-50:])
        print(a['code'], b['code'], t, p)
