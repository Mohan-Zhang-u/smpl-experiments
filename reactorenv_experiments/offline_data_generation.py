import numpy as np
import yaml
import pickle
from mzutils import get_things_in_loc, mkdir_p
import random

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *


STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
initial_state_deviation_ratio_t=0.15
generate_initial_states = True
load_full_states = True
npseparate_initial_states = False
generate_algorithmatic_dataset = True
dataset_dir = 'datasets'
curr_model_name = 'mpc' # 'pid'




def separate_initial_states(initial_states, checkfunc, checkfuncname='suit'):
    suit_name = f'datasets/initial_states_step_{checkfuncname}.npy'
    not_suit_name = f'datasets/initial_states_step_not_{checkfuncname}.npy'
    suit_initial_states = []
    not_suit_initial_states = []
    for initial_state in initial_states:
        if checkfunc(initial_state):
            suit_initial_states.append(initial_state)
        else:
            not_suit_initial_states.append(initial_state)
    np.save(suit_name, np.array(suit_initial_states))
    np.save(not_suit_name, np.array(not_suit_initial_states))
    return suit_initial_states
    
def check_all_below(initial_states):
    steady_observations = np.array(STEADY_OBSERVATIONS)
    if (initial_states < steady_observations).all():
        return True
    else:
        return False
    
def check_all_above(initial_states):
    steady_observations = np.array(STEADY_OBSERVATIONS)
    if (initial_states > steady_observations).all():
        return True
    else:
        return False

def check_all_in_range(initial_states):
    steady_observations = np.array(STEADY_OBSERVATIONS)
    low=(1-initial_state_deviation_ratio_t)*steady_observations
    high=(1+initial_state_deviation_ratio_t)*steady_observations
    if (initial_states > low).all() and (initial_states < high).all():
        return True
    else:
        return False
    
def check_all_out_of_range(initial_states):
    steady_observations = np.array(STEADY_OBSERVATIONS)
    low=(1-initial_state_deviation_ratio_t)*steady_observations
    high=(1+initial_state_deviation_ratio_t)*steady_observations
    if (initial_states < low).all() or (initial_states > high).all():
        return True
    else:
        return False

if __name__ == "__main__":
    with open(f'offline_experiments.yaml', 'r') as fp:
        config_dict = yaml.safe_load(fp)
    env_name = config_dict['env_name']
    model_name = config_dict['model_name']
    plt_dir = config_dict['plt_dir']
    normalize = config_dict['normalize']
    compute_diffs_on_reward = config_dict['compute_diffs_on_reward']
    dense_reward = config_dict['dense_reward']
    debug_mode = config_dict['debug_mode']
    dataset_location = config_dict['dataset_location']
    seed = config_dict['seed']

    np.random.seed(seed)
    random.seed(seed)

    # init env
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
    
    val_per_states = [7, 10, 20, 30, 40, 50, 60]
    for val_per_state in val_per_states:
        env.reset()
        
        
        if generate_initial_states:
            mkdir_p('datasets')
            env.evenly_spread_initial_states(val_per_state, dump_location=f'datasets/initial_states_step_{val_per_state}.npy')
            
        if load_full_states:
            num_episodes = val_per_state**3
            initial_states = np.load(f'datasets/initial_states_step_{val_per_state}.npy')

        if npseparate_initial_states:
            initial_states = np.load(f'datasets/initial_states_step_{val_per_state}.npy')
            sep_toks = [(check_all_below, f'{val_per_state}_below'), (check_all_above, f'{val_per_state}_above'), (check_all_in_range, f'{val_per_state}_in_range'), (check_all_out_of_range, f'{val_per_state}_out_of_range')]
            for sep_tok in sep_toks:
                initial_states = separate_initial_states(initial_states, sep_tok[0], sep_tok[1])
    

    if generate_algorithmatic_dataset:
        initial_states_loc_lst = get_things_in_loc(dataset_dir)
        initial_states_loc_lst.sort()
        for initial_states_loc in initial_states_loc_lst:
            initial_states = np.load(initial_states_loc)
            if curr_model_name == 'pid':
                from smpl.envs.reactorenv import ReactorPID
                curr_model = ReactorPID(Kis=[100.0, 0.5], steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05], max_action=[35.0, 0.2])
            elif curr_model_name == 'mpc':
                from smpl.envs.reactorenv import ReactorMPC
                curr_model = ReactorMPC()
            else:
                raise ValueError('curr_model_name must be pid or mpc')
            print(initial_states_loc)
            print('length', len(initial_states))
            dataset = env.generate_dataset_with_algorithm(curr_model, normalize=normalize, num_episodes=len(initial_states), initial_states=initial_states, format='d4rl')
            print("terminals", np.where(dataset["terminals"] == True))
            print("timeouts", np.where(dataset["timeouts"] == True))
            name = initial_states_loc.replace('initial_states', curr_model_name)
            name = name.replace('.npy', '')
            with open(f"{name}_normalize={normalize}.pkl", 'wb') as fp:
                pickle.dump(dataset, fp)
            print(f"saved dataset {name}_normalize={normalize}")
