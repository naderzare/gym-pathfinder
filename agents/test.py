import argparse
import time


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
parser.add_argument('-r', '--rl_rotating', help='using rotating in training', type=str2bool, default=False)
parser.add_argument('-t', '--test', help='just test', type=str2bool, default=False)
parser.add_argument('-tr', '--test_rotating', help='test_rotating', type=str2bool, default=True)
parser.add_argument('-uh', '--use_her', help='using HER', type=str2bool)
parser.add_argument('-ht', '--her_type', help='HER Type', type=str, default='future')
parser.add_argument('-hn', '--her_number', help='HER Number', type=int, default=4)
parser.add_argument('-n', '--name', help='Run Name', type=str, default='test_'+str(time.time()))
parser.add_argument('-map', '--map', help='Map Path', type=str, default='/home/nader/workspace/rl/gym-pathfinder/agents/maps/vertical_map/')
args = parser.parse_args()
print(args)
exit()