import d3rlpy
import json

import numpy as np
import pandas as pd
import random
import mzutils
from zipfile import ZipFile
from gym import spaces, Env
import wandb
import yaml
import pickle
import shutil
from datetime import datetime
import re

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *

with open(f'offline_experiments.yaml', 'r') as fp:
    config_dict = yaml.safe_load(fp)
seed = config_dict['seed']
num_of_seeds = config_dict['num_of_seeds']
env_name = config_dict['env_name']
model_name = config_dict['model_name']
normalize = config_dict['normalize']
dense_reward = config_dict['dense_reward']
debug_mode = config_dict['debug_mode']

# for offlineRL online learning
online_training = config_dict['online_training']
buffer_maxlen = config_dict['buffer_maxlen']
explorer_start_epsilon = config_dict['explorer_start_epsilon']
explorer_end_epsilon = config_dict['explorer_end_epsilon']
explorer_duration = config_dict['explorer_duration']
n_steps_per_epoch = config_dict['n_steps_per_epoch']
online_random_steps = config_dict['online_random_steps']
online_update_interval = config_dict['online_update_interval']
online_save_interval = config_dict['online_save_interval']

# for offline data generation and training
N_EPOCHS  = config_dict['N_EPOCHS']
DYNAMICS_N_EPOCHS = config_dict['DYNAMICS_N_EPOCHS']
scaler = config_dict['scaler']
action_scaler = config_dict['action_scaler']
reward_scaler = config_dict['reward_scaler']
evaluate_on_environment = config_dict['evaluate_on_environment']
default_loc = config_dict['default_loc']
plt_dir = config_dict['plt_dir']
dataset_location = config_dict['dataset_location']
training_dataset_loc = config_dict['training_dataset_loc']
eval_dataset_loc = config_dict['eval_dataset_loc']
test_initial_states = config_dict['test_initial_states']

# env specific configs
reward_on_steady = config_dict.get('reward_on_steady', None)
reward_on_absolute_efactor = config_dict.get('reward_on_absolute_efactor', None)
compute_diffs_on_reward = config_dict.get('compute_diffs_on_reward', None)
standard_reward_style = config_dict.get('standard_reward_style', None)
initial_state_deviation_ratio = config_dict.get('initial_state_deviation_ratio', None)

if seed is not None:
    seeds = [seed]
else:
    num_of_seeds = config_dict['num_of_seeds']
    seeds = []
    for i in range(num_of_seeds):
        seeds.append(random.randint(0, 2**32-1))

# init env
env_config = set_env_config(env_name, normalize=normalize, dense_reward=dense_reward,
                            reward_on_steady=reward_on_steady,
                            reward_on_absolute_efactor=reward_on_absolute_efactor,
                            compute_diffs_on_reward=compute_diffs_on_reward,
                            standard_reward_style=standard_reward_style,
                            initial_state_deviation_ratio=initial_state_deviation_ratio)
env = env_creator(env_config)
env.reset()
    


# from smpl.envs.mabenv import MAbUpstreamEMPC
# baseline_model_name = 'empc'
# baseline_model = MAbUpstreamEMPC(env.controller)
from smpl.envs.mabenv import MAbUpstreamMPC
baseline_model_name = 'baseline'
baseline_model = MAbUpstreamMPC(env.controller)

algo_names = ['baseline', 'BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO']
for algo_name in algo_names:
    num_episodes = 400
    initial_states = np.load(test_initial_states)
    if algo_name == 'baseline':
        algorithms = [(baseline_model, baseline_model_name, normalize)]
    else:
        curr_algo = OfflineRLModel(algo_name, config_dict_loc=f'OFFLINE_BEST.yaml')
        # algorithms = [(curr_algo, algo_name, normalize), (baseline_model, baseline_model_name, normalize)]
        algorithms = [(curr_algo, algo_name, normalize)]
    dataset = env.generate_dataset_with_algorithm(algorithms[0][0], normalize=False, num_episodes=2, format='d4rl')
    print(algo_name)
    print(len(dataset['predict_time_taken']))
    print(np.mean(dataset['predict_time_taken']))
    print('\n')
    
    


