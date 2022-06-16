import argparse
import codecs
import csv
import os
import sys

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def env_creator(env_config):
    """
    so that all environments are created in the same way, in training and inference.
    has to be in online_experiments, otherwise will trigger ModuleNotFoundError: No module named 'models'
    in ray/serialization.py
    """

    if env_config["env_name"] == 'pensimenv':
        from pensimpy.examples.recipe import Recipe, RecipeCombo
        from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
        from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
            PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE
        from smpl.envs.pensimenv import PenSimEnvGym
        recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                       FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                       FG: Recipe(FG_DEFAULT_PROFILE, FG),
                       PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                       DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                       WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                       PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}
        recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
        # set up the environment
        env = PenSimEnvGym(recipe_combo=recipe_combo, normalize=env_config["normalize"],
                           dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'beerfmtenv':
        from smpl.envs.beerfmtenv import BeerFMTEnvGym
        env = BeerFMTEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'atropineenv':
        from smpl.envs.atropineenv import AtropineEnvGym
        env = AtropineEnvGym(normalize=env_config["normalize"],
                             reward_scaler=100000)  # by default uses reward on steady.
    elif env_config["env_name"] == 'reactorenv':
        from smpl.envs.reactorenv import ReactorEnvGym
        env = ReactorEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"],
                            compute_diffs_on_reward=env_config["compute_diffs_on_reward"])
    elif env_config["env_name"] == 'mabupstreamenv':
        from smpl.envs.mabenv import MAbUpstreamEnvGym
        env = MAbUpstreamEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'mabenv':
        from smpl.envs.mabenv import MAbEnvGym
        env = MAbEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"],
                        standard_reward_style=env_config["standard_reward_style"],
                        initial_state_deviation_ratio=env_config["initial_state_deviation_ratio"])
    else:
        raise ValueError('env_name not recognized')
    return env


register_env("my_env", env_creator)

with open(f'online_experiments.yaml', 'r') as fp:
    config_dict = yaml.safe_load(fp)
env_name = config_dict['env_name']
model_name = config_dict['model_name']
normalize = config_dict['normalize']
dense_reward = config_dict['dense_reward']
debug_mode = config_dict['debug_mode']

scheduler_name = config_dict['scheduler_name']
time_budget_s = config_dict['time_budget_s']
use_tune = config_dict['use_tune']
num_gpus = config_dict['num_gpus']
num_workers = config_dict['num_workers']
# num_gpus_per_worker = config_dict['num_gpus_per_worker']
train_iter = config_dict['train_iter']
plt_dir = config_dict['plt_dir']
local_dir = config_dict['local_dir']
log_to_file = config_dict['log_to_file']

best_checkpoint_path = config_dict.get('best_checkpoint_path', None)

# env specific configs
reward_on_steady = config_dict.get('reward_on_steady', None)
reward_on_absolute_efactor = config_dict.get('reward_on_absolute_efactor', None)
compute_diffs_on_reward = config_dict.get('compute_diffs_on_reward', None)
standard_reward_style = config_dict.get('standard_reward_style', None)
initial_state_deviation_ratio = config_dict.get('initial_state_deviation_ratio', None)

# eval specific configs
eval_num_episodes = config_dict.get('eval_num_episodes', 100)

# use argparse to overwrite
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument("--normalize", type=str, default='')
args = parser.parse_args()
if args.env_name != '':
    env_name = args.env_name
if args.model_name != '':
    model_name = args.model_name
if args.normalize != '':
    s = args.normalize
    if s.lower() in ('yes', 'y', 't', 'true', 1):
        normalize = True
    elif s.lower() in ('no', 'n', 'f', 'false', 0):
        normalize = False
    else:
        normalize = None
local_dir = os.path.join(local_dir, env_name, model_name)
project_name = local_dir.replace('/', '_')

env_config = set_env_config(env_name, normalize=normalize, dense_reward=dense_reward,
                            reward_on_steady=reward_on_steady,
                            reward_on_absolute_efactor=reward_on_absolute_efactor,
                            compute_diffs_on_reward=compute_diffs_on_reward,
                            standard_reward_style=standard_reward_style,
                            initial_state_deviation_ratio=initial_state_deviation_ratio)
env = env_creator(env_config)
env.reset()

model_names = [
    'a3c',
    'ars',
    'impala',
    'pg',
    'ppo',
    'sac',
    'ddpg',

]
best_checkpoint_paths = [
    # the best checkpoint location. Fill with your own. For example:
    # 'atropine_online_results/atropineenv/a3c/A3CTrainer/A3CTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/ars/ARSTrainer/ARSTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/impala/ImpalaTrainer/ImpalaTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/pg/PGTrainer/PGTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/ppo/PPOTrainer/PPOTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/sac/SACTrainer/SACTrainer_my_env/checkpoint_002000/checkpoint-2000',
    # 'atropine_online_results/atropineenv/ddpg/DDPGTrainer/DDPGTrainer_my_env/checkpoint_002000/checkpoint-2000',
]
config_dirs = [
    # the config location of your best checkpoint. Fill with your own. For example:
    # 'atropine_online_results/atropineenv/a3c/A3CTrainer/A3CTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/ars/ARSTrainer/ARSTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/impala/ImpalaTrainer/ImpalaTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/pg/PGTrainer/PGTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/ppo/PPOTrainer/PPOTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/sac/SACTrainer/SACTrainer_my_env/params.pkl',
    # 'atropine_online_results/atropineenv/ddpg/DDPGTrainer/DDPGTrainer_my_env/params.pkl',
]
results_csv = ['algo_name', 'on_episodes_reward_mean', 'episodes_reward_std', 'all_reward_mean', 'all_reward_std']

try:
    for i in range(len(model_names)):
        algo_name = model_names[i]
        best_checkpoint_path = best_checkpoint_paths[i]
        config_dir = config_dirs[i]
        num_episodes = eval_num_episodes
        curr_algo = OnlineRLModel(model_name=algo_name, ckpt_path=best_checkpoint_path, env="my_env",
                                  config_dir=config_dir, num_gpus=0)
        algorithms = [(curr_algo, algo_name, normalize)]
        save_dir = os.path.join(plt_dir, algo_name)
        observations_list, actions_list, rewards_list = env.evalute_algorithms(algorithms, num_episodes=num_episodes,
                                                                               to_plt=False, plot_dir=save_dir)
        # or, if you also generated an eval dataset, you can use the following code:
        # eval_initial_states = np.load(eval_initial_states_loc)
        # observations_list, actions_list, rewards_list = env.evalute_algorithms(algorithms, num_episodes=num_episodes, initial_states=eval_initial_states, to_plt=False, plot_dir=save_dir)
        results_dict = env.report_rewards(rewards_list, algo_names=env.algorithms_to_algo_names(algorithms),
                                          save_dir=save_dir)
        results_csv.append([algo_name, results_dict[f'{algo_name}_on_episodes_reward_mean'],
                            results_dict[f'{algo_name}_on_episodes_reward_std'],
                            results_dict[f'{algo_name}_all_reward_mean'], results_dict[f'{algo_name}_all_reward_std']])
        np.save(os.path.join(save_dir, f'observations.npy'), observations_list)
        np.save(os.path.join(save_dir, f'actions.npy'), actions_list)
        np.save(os.path.join(save_dir, f'rewards.npy'), rewards_list)
except FileNotFoundError as e:
    print(e)
with codecs.open(os.path.join(plt_dir, "total_results_dict.csv"), "w+", encoding="utf-8") as fp:
    csv_writer = csv.writer(fp)
    for row in results_csv:
        csv_writer.writerow(row)
