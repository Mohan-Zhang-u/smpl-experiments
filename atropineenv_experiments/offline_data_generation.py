import numpy as np
import random
import json
import yaml
import os
import sys
import argparse
import pickle
import mzutils
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *


if __name__ == "__main__":
    with open(f'offline_experiments.yaml', 'r') as fp:
        config_dict = yaml.safe_load(fp)
        env_name = config_dict['env_name']
        model_name = config_dict['model_name']
        normalize = config_dict['normalize']
        dense_reward = config_dict['dense_reward']
        
        dataset_location = config_dict['dataset_location']
        seed = config_dict['seed']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--initial_states_loc', type=str, default='')
    parser.add_argument('--num_of_initial_state', type=int, default=1000)
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
        
        
    np.random.seed(seed)
    random.seed(seed)

    # init env
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
    
    # initial states
    if args.initial_states_loc == '':
        num_of_initial_state = args.num_of_initial_state
        initial_states = []
        for i in tqdm(range(args.num_of_initial_state)):
            init = env.sample_initial_state()
            initial_states.append(init)
        initial_states = np.array(initial_states)
        initial_states_loc = os.path.join(dataset_location, f'{env_name}/initial_states')
        mzutils.mkdir_p(initial_states_loc)
        initial_states_loc = os.path.join(initial_states_loc, f'{num_of_initial_state}.npy')
        np.save(initial_states_loc, initial_states)
    else:
        initial_states_loc = args.initial_states_loc
        initial_states = np.load(initial_states_loc)
    

    from smpl.envs.atropineenv import AtropineMPC
    model_name = 'empc'
    curr_model = AtropineMPC()
    print(initial_states_loc)
    print('length', len(initial_states))
    dataset = env.generate_dataset_with_algorithm(curr_model, normalize=normalize, num_episodes=len(initial_states), initial_states=initial_states, format='d4rl')
    dataset_loc = os.path.join(dataset_location, f'{env_name}/{model_name}')
    mzutils.mkdir_p(dataset_loc)
    dataset_loc = os.path.join(dataset_loc, f'{num_of_initial_state}_normalize={normalize}.pkl')
    print("terminals", np.where(dataset["terminals"] == True))
    print("timeouts", np.where(dataset["timeouts"] == True))
    with open(dataset_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {dataset_loc}")
