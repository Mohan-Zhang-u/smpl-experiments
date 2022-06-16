import d3rlpy
import json
import yaml
import os
import sys
import argparse
import pickle
import mzutils
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import copy
# from d3rlpy.gpu import Device
import mzutils
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, \
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE
from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData, NUM_STEPS


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
        standard_reward_style = config_dict.get('standard_reward_style', None)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--initial_states_loc', type=str, default='')
    parser.add_argument('--num_of_initial_state', type=int, default=400)
    parser.add_argument('--evenly_spread_initial_states', type=mzutils.argparse_bool, default=False)
    parser.add_argument('--val_per_state', type=int, default=10)
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


    recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                FG: Recipe(FG_DEFAULT_PROFILE, FG),
                PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}

    recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
    # set up the environment
    env = PenSimEnvGym(recipe_combo=recipe_combo)
    env.seed(seed)
    dataset_obj = PeniControlData(dataset_folder='pensimpy_1010_samples', normalize=normalize)
    if dataset_obj.file_list:
        print('Penicillin_Control_Challenge data correctly initialized.')
    else:
        raise ValueError("Penicillin_Control_Challenge data initialization failed.")
    file_list = dataset_obj.file_list
    number_of_training_set = 900
    training_items = random.sample(file_list, number_of_training_set)
    evaluating_items = copy.deepcopy(file_list)
    for i in training_items:
        evaluating_items.remove(i)
    dataset_obj.file_list = training_items
    dataset_d4rl_training = dataset_obj.get_dataset()
    dataset_obj.file_list = evaluating_items
    dataset_d4rl_evaluating = dataset_obj.get_dataset()
    dataset_loc = os.path.join(dataset_location, f'{env_name}')
    mzutils.mkdir_p(dataset_loc)
    
    dataset = dataset_d4rl_training
    tmp_dataset_loc = os.path.join(dataset_loc, f'{number_of_training_set}_normalize={normalize}.pkl')
    with open(tmp_dataset_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {tmp_dataset_loc}")
    
    dataset = dataset_d4rl_evaluating
    tmp_dataset_loc = os.path.join(dataset_loc, f'{1010-number_of_training_set}_normalize={normalize}.pkl')
    with open(tmp_dataset_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {tmp_dataset_loc}")
