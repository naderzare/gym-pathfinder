import matplotlib.pyplot as plt
import os
import matplotlib

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

path = 'results/res'
dirs = os.listdir(path)
dirs.sort(key=natural_keys)
avg_number = 1
# leg = ['min','max_move_neg','max_move_neg_g0','max_move_neg_g0_4','max_sparse','max_sparse_g0']
# colors = ['darkorange', 'crimson', 'royalblue', 'black']
all_dir = []
fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0})
fig.set_figheight(4)
fig.set_figwidth(15)
import re
for d in range(len(dirs)):
    # if dirs[d] not in need_dirs:
    #     continue
    dir_name = dirs[d]
    if dir_name.find('simple') != -1:
        continue
    # if dir_name.find('max') == -1:
    #     continue
    # if re.match('(t[1234]_)|(t43_)+', dir_name):
    #     continue
    all_dir.append(dirs[d])
    fh_1 = open(os.path.join(path, dirs[d], 'horizontal_difqr.txt')).readlines()
    fv_1 = open(os.path.join(path, dirs[d], 'vertical_difqr.txt')).readlines()
    fd_1 = open(os.path.join(path, dirs[d], 'diagonal_difqr.txt')).readlines()
    acc_h_1 = []
    acc_v_1 = []
    acc_d_1 = []
    for i in range(len(fh_1)):
        acc_h_1.append(float(fh_1[i]))
        acc_v_1.append(float(fv_1[i]))
        acc_d_1.append(float(fd_1[i]))
    acc_avg_h_1 = [acc_h_1[0]]
    acc_avg_v_1 = [acc_v_1[0]]
    acc_avg_d_1 = [acc_d_1[0]]
    for i in range(int(len(acc_h_1) / avg_number)):
        start = i * avg_number + 1
        end = (i + 1) * avg_number + 1
        if (i + 1) * avg_number > len(acc_h_1):
            break
        acc_avg_h_1.append(sum(acc_h_1[start:end]) / avg_number)
        acc_avg_v_1.append(sum(acc_v_1[start:end]) / avg_number)
        acc_avg_d_1.append(sum(acc_d_1[start:end]) / avg_number)
    print(dir_name, max(acc_h_1), max(acc_v_1), max(acc_d_1))
    axes[0].plot(acc_avg_h_1, linewidth=2)
    axes[1].plot(acc_avg_v_1, linewidth=2)
    axes[2].plot(acc_avg_d_1, linewidth=2)
    axes[0].set_title('H')
    axes[1].set_title('V')
    axes[2].set_title('D')
    for ax in axes.flat:
        # ax.set_xlim([0,20])
        # ax.set_ylim([0, 100000])
        ax.set(xlabel='epochs')
    # plt.fill_between(range(0,201), acc1, acc2, color=colors[d], alpha=0.4)
import matplotlib.patches as patches

# r1 = patches.Rectangle((0,0),1,1,facecolor=colors[0])
# r2 = patches.Rectangle((0,0),1,1,facecolor=colors[1])
# r3 = patches.Rectangle((0,0),1,1,facecolor=colors[2])
# r4 = patches.Rectangle((0,0),1,1,facecolor=colors[3])
plt.legend(all_dir,bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.legend((r1,r2,r3, r4), leg)
plt.show()