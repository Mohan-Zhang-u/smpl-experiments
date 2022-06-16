import argparse

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *


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
import ray.rllib.agents.ppo as ppo

algo_config_dict_location = 'ray_results/beerfmtenv/ppo/PPOTrainer/PPOTrainer_my_env/params.pkl'
with open(algo_config_dict_location, 'rb') as fp:
    algo_config_dict = pickle.load(fp)
agent = ppo.PPOTrainer(config=algo_config_dict, env="my_env")
agent.restore('ray_results/beerfmtenv/ppo/PPOTrainer/PPOTrainer_my_env/checkpoint_002000/checkpoint-2000')

# instantiate env class


# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
