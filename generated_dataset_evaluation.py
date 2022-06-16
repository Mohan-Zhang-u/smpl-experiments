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
N_EPOCHS = config_dict['N_EPOCHS']
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

env_config = set_env_config(env_name, normalize=normalize, dense_reward=dense_reward,
                            reward_on_steady=reward_on_steady,
                            reward_on_absolute_efactor=reward_on_absolute_efactor,
                            compute_diffs_on_reward=compute_diffs_on_reward,
                            standard_reward_style=standard_reward_style,
                            initial_state_deviation_ratio=initial_state_deviation_ratio)
env = env_creator(env_config)
env.reset()

dataset_loc = 'mabenv2_experiments/offline_datasets/mabenv/empc/100_normalize=False'
with open(f'{dataset_loc}.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

observations_list, actions_list, rewards_list = env.dataset_to_observations_actions_rewards_list(dataset)
env.report_rewards(rewards_list, save_dir=dataset_loc)
print(1)
