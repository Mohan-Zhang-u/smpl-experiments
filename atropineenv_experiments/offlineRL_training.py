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


class SeedData:
    """
    A dictionary that aims to average the evaluated mean_episode_return accross different random seed.
    Also controls where to resume the experiments from.
    """

    def __init__(self, save_path, resume_from={}):
        self.seed_data = pd.DataFrame({
            'algo_name': pd.Series([], dtype='str'),
            'test_reward': pd.Series([], dtype='float'),
            'seed': pd.Series([], dtype='int'),
        })
        self.save_path = save_path
        mzutils.mkdir_p(save_path)
        self.load()
        # set experiment range
        self.resume_from = resume_from
        self.resume_check_passed = False

    def load(self):
        re_list = mzutils.get_things_in_loc(self.save_path)
        if not re_list:
            print("Cannot find the a seed_data.csv at", self.save_path, "initializing a new one.")
            self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)
        else:
            self.seed_data = pd.read_csv(os.path.join(self.save_path, 'seed_data.csv'))
            print("Loaded the seed_data.csv at", self.save_path)
    
    def save(self):
        self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)

    def append(self, algo_name, test_reward, seed):
        self.seed_data.loc[len(self.seed_data)] = [algo_name, test_reward, seed]

    def setter(self, algo_name, test_reward, seed):
        # average over seed makes seed==-1
        # online makes dataset_percent==0.0
        self.append(algo_name, test_reward, seed)
        averaged_reward = self.seed_data.loc[(self.seed_data['algo_name'] == algo_name)]['test_reward'].mean()
        if seed == seeds[-1]: # append the average, seed now set to -1
            self.seed_data.loc[len(self.seed_data)] = [algo_name, averaged_reward, -1]
        self.save()
        return averaged_reward

    def resume_checker(self, current_positions):
        """
        current_positions has the same shape as self.resume_from
        return True if the current loop still need to be skipped.
        """
        if self.resume_check_passed is True: # checker has already passed.
            return True

        if not self.resume_from:
            self.resume_check_passed = True
        elif all([self.resume_from[condition] is None for condition in self.resume_from]):
            self.resume_check_passed = True
        else:
            self.resume_check_passed = all([self.resume_from[condition] == current_positions[condition] for condition in self.resume_from])
        return self.resume_check_passed


with open('offline_experiments.yaml', 'r') as fp:
    config_dict = yaml.safe_load(fp)
env_name = config_dict['env_name']
model_name = config_dict['model_name']
dense_reward = config_dict['dense_reward']
default_loc = config_dict['default_loc']
plt_dir = config_dict['plt_dir']
training_dataset_loc = config_dict['training_dataset_loc']
eval_dataset_loc = config_dict['eval_dataset_loc']
N_EPOCHS = config_dict['N_EPOCHS']
DYNAMICS_N_EPOCHS = config_dict['DYNAMICS_N_EPOCHS']
normalize = config_dict['normalize']
scaler = config_dict['scaler']
action_scaler = config_dict['action_scaler']
reward_scaler = config_dict['reward_scaler']
evaluate_on_environment = config_dict['evaluate_on_environment']
debug_mode = config_dict['debug_mode']
seed = config_dict.get('seed', None)
if seed is not None:
    seeds = [seed]
else:
    num_of_seeds = config_dict['num_of_seeds']
    seeds = []
    for i in range(num_of_seeds):
        seeds.append(random.randint(0, 2**32-1))

reward_on_steady = config_dict.get('reward_on_steady', None)
reward_on_absolute_efactor = config_dict.get('reward_on_absolute_efactor', None)
compute_diffs_on_reward = config_dict.get('compute_diffs_on_reward', None)
standard_reward_style = config_dict.get('standard_reward_style', None)
initial_state_deviation_ratio = config_dict.get('initial_state_deviation_ratio', None)
env_config = set_env_config(env_name, normalize=normalize, dense_reward=dense_reward,
                        reward_on_steady=reward_on_steady,
                        reward_on_absolute_efactor=reward_on_absolute_efactor,
                        compute_diffs_on_reward=compute_diffs_on_reward,
                        standard_reward_style=standard_reward_style,
                        initial_state_deviation_ratio=initial_state_deviation_ratio)
env = env_creator(env_config)

env.reset()

with open(training_dataset_loc, 'rb') as handle:
    training_dataset_pkl = pickle.load(handle)
with open(eval_dataset_loc, 'rb') as handle:
    eval_dataset_pkl = pickle.load(handle)


algo_names = ['CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO', 'BC']
resume_from = {
    "seed": None,
    "dataset_name": None,
    "algo_name": None,
}

with open('project_title.txt' , 'r+') as f:
    project_title = f.readline()
    int_in_title = re.search(r'\d+', project_title).group()
    project_title = project_title.replace(int_in_title, str(int(int_in_title)+1))
with open('project_title.txt' , 'w') as f:
    f.write(project_title)
if debug_mode:
    project_title = 'DEBUG_' + project_title

seed_data = SeedData(save_path=default_loc, resume_from=resume_from)
for seed in seeds:
    # set random seeds in random module, numpy module and PyTorch module.
    d3rlpy.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dataset = d3rlpy.dataset.MDPDataset(training_dataset_pkl['observations'], training_dataset_pkl['actions'], training_dataset_pkl['rewards'], training_dataset_pkl['terminals'])
    eval_dataset = d3rlpy.dataset.MDPDataset(eval_dataset_pkl['observations'], eval_dataset_pkl['actions'], eval_dataset_pkl['rewards'], eval_dataset_pkl['terminals'])
    feeded_episodes = dataset.episodes
    eval_feeded_episodes = eval_dataset.episodes
    for algo_name in algo_names:
        # ---------------- check for resume, should be put at the start of of most inner loop ----------------
        current_positions = {
            "seed": seed,
            "algo_name": algo_name,
        }
        prev_evaluate_on_environment_scorer = float('-inf')
        prev_continuous_action_diff_scorer = float('inf')
        if not seed_data.resume_checker(current_positions):
            continue
        # ---------------- check for resume, should be put at the start of of most inner loop ----------------
        if algo_name == 'CQL':
            curr_algo = d3rlpy.algos.CQL(q_func_factory='qr', use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler) # use Quantile Regression Q function, default was 'mean'

        elif algo_name == 'PLAS':
            curr_algo = d3rlpy.algos.PLAS(q_func_factory='qr', use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler) # use Quantile Regression Q function, default was 'mean'
        elif algo_name == 'PLASWithPerturbation':
            curr_algo = d3rlpy.algos.PLASWithPerturbation(q_func_factory='qr', use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler) # use Quantile Regression Q function, default was 'mean'
        elif algo_name == 'DDPG':
            curr_algo = d3rlpy.algos.DDPG(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'BC':
            curr_algo = d3rlpy.algos.BC(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'TD3':
            curr_algo = d3rlpy.algos.TD3(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'BEAR':
            curr_algo = d3rlpy.algos.BEAR(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'SAC':
            curr_algo = d3rlpy.algos.SAC(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'BCQ':
            curr_algo = d3rlpy.algos.BCQ(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'CRR':
            curr_algo = d3rlpy.algos.CRR(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'AWR':
            curr_algo = d3rlpy.algos.AWR(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'AWAC':
            curr_algo = d3rlpy.algos.AWAC(use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
        elif algo_name == 'COMBO':
            dynamics = 1
        #     dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)
        #     curr_algo = d3rlpy.algos.COMBO(use_gpu=True)
        elif algo_name == 'MOPO':
            dynamics = 1
        #     dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)
        #     curr_algo = d3rlpy.algos.MOPO(use_gpu=True)
        else:
            raise Exception("algo_name is invalid!", algo_name)
        # print(dataset_name, env.action_space.shape, env.observation_space.shape, len(dataset.episodes), np.ceil(len(dataset.episodes)*0.01))
        
        logdir = default_loc+str(seed)
        acutal_dir = logdir+'/'+algo_name
        wandb_run = wandb.init(reinit=True, project=project_title, name=acutal_dir, dir=default_loc)
        # --------- Model Based Algorithms leverages the probablistic ensemble dynamics model to generate new dynamics data with uncertainty penalties.  --------- 
        if algo_name in ['COMBO', 'MOPO']:
            scorers={
                'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
                'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
                'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
            }
            # train dynamics model first
            dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
            dynamics.fit(feeded_episodes,
                eval_episodes=eval_feeded_episodes,
                n_epochs=DYNAMICS_N_EPOCHS,
                logdir=logdir,
                scorers=scorers)
            if algo_name == 'COMBO':
                curr_algo = d3rlpy.algos.COMBO(dynamics=dynamics, use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'MOPO':
                curr_algo = d3rlpy.algos.MOPO(dynamics=dynamics, use_gpu=True, action_scaler=action_scaler, reward_scaler=reward_scaler)
            else:
                raise Exception("algo_name is invalid!")
        # --------- Model Based Algorithms leverages the probablistic ensemble dynamics model to generate new dynamics data with uncertainty penalties.  --------- 
        
        if algo_name == 'BC':
            scorers = {
                'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            }
        elif algo_name == 'AWR':
            scorers = {
                'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                # 'value_estimation_std_scorer': d3rlpy.metrics.scorer.value_estimation_std_scorer,
                'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            }
        else:
            scorers = {
                'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                'value_estimation_std_scorer': d3rlpy.metrics.scorer.value_estimation_std_scorer,
                'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            }
            
        if evaluate_on_environment:
            scorers['evaluate_on_environment_scorer'] = d3rlpy.metrics.scorer.evaluate_on_environment(env)
        
        for epoch, metrics in curr_algo.fitter(feeded_episodes, eval_episodes=eval_feeded_episodes, n_epochs=N_EPOCHS, with_timestamp=False, logdir=logdir, scorers=scorers):
            wandb.log(metrics)
            if evaluate_on_environment:
                if metrics['evaluate_on_environment_scorer'] > prev_evaluate_on_environment_scorer:
                    prev_evaluate_on_environment_scorer = metrics['evaluate_on_environment_scorer']
                    curr_algo.save_model(os.path.join(acutal_dir, 'best_evaluate_on_environment_scorer.pt'))
            if metrics['continuous_action_diff_scorer'] < prev_continuous_action_diff_scorer:
                prev_continuous_action_diff_scorer = metrics['continuous_action_diff_scorer']
                curr_algo.save_model(os.path.join(acutal_dir, 'best_continuous_action_diff_scorer.pt'))
        if evaluate_on_environment:
            shutil.copyfile(os.path.join(acutal_dir, 'best_evaluate_on_environment_scorer.pt'), os.path.join(acutal_dir, 'best.pt'))
        else:
            shutil.copyfile(os.path.join(acutal_dir, 'best_continuous_action_diff_scorer.pt'), os.path.join(acutal_dir, 'best.pt'))
        wandb_run.finish()
