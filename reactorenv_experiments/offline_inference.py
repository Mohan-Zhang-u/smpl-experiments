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
import csv
import codecs
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
    

from smpl.envs.reactorenv import ReactorMPC
from smpl.envs.mabenv import MAbUpstreamEMPC
baseline_model_name = 'baseline'
baseline_model = ReactorMPC()

algo_names = ['baseline', 'BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO']
results_csv = ['algo_name', 'on_episodes_reward_mean', 'episodes_reward_std', 'all_reward_mean', 'all_reward_std']
try:
    for algo_name in algo_names:
        initial_states = np.load(test_initial_states)
        num_episodes = len(initial_states)
        if algo_name == 'baseline':
            algorithms = [(baseline_model, baseline_model_name, normalize)]
        else:
            curr_algo = OfflineRLModel(algo_name, config_dict_loc=f'OFFLINE_BEST.yaml')
            # algorithms = [(curr_algo, algo_name, normalize), (baseline_model, baseline_model_name, normalize)]
            algorithms = [(curr_algo, algo_name, normalize)]
        # in atropine env, initial_states does not contain all information of the initial condition of the environment configuration (zk, xk not included.)
        save_dir = os.path.join(plt_dir, algo_name)
        observations_list, actions_list, rewards_list = env.evalute_algorithms(algorithms, num_episodes=num_episodes, initial_states=None, to_plt=False, plot_dir=save_dir)
        results_dict = env.report_rewards(rewards_list, algo_names=env.algorithms_to_algo_names(algorithms), save_dir=save_dir)
        results_csv.append([algo_name, results_dict[f'{algo_name}_on_episodes_reward_mean'], results_dict[f'{algo_name}_on_episodes_reward_std'], results_dict[f'{algo_name}_all_reward_mean'], results_dict[f'{algo_name}_all_reward_std']])
        np.save(os.path.join(save_dir, f'observations.npy'), observations_list)
        np.save(os.path.join(save_dir, f'actions.npy'), actions_list)
        np.save(os.path.join(save_dir, f'rewards.npy'), rewards_list)
except FileNotFoundError as e:
    print(e)
with codecs.open(os.path.join(plt_dir, "total_results_dict.csv"), "w+", encoding="utf-8") as fp:
    csv_writer = csv.writer(fp)
    for row in results_csv:
        csv_writer.writerow(row)

# mzutils.mkdir_p(eval_results_loc)
# with open(os.path.join(observations_list, f'{algo_name}_observations.pkl'), 'wb') as fp:
#     pickle.dump(observations_list, fp)
# with open(os.path.join(actions_list, f'{algo_name}_actions.pkl'), 'wb') as fp:
#     pickle.dump(actions_list, fp)
# with open(os.path.join(rewards_list, f'{algo_name}_rewards.pkl'), 'wb') as fp:
#     pickle.dump(rewards_list, fp)
    


