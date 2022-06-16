import numpy as np
import json
import yaml
import argparse

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models import *

from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open(f'online_experiments.yaml', 'r') as fp:
        config_dict = yaml.safe_load(fp)
        env_name = config_dict['env_name']
        model_name = config_dict['model_name']
        normalize = config_dict['normalize']
        dense_reward = config_dict['dense_reward']
        
        best_checkpoint_path = config_dict.get('best_checkpoint_path', None)
        use_tune = config_dict['use_tune']
        num_gpus = config_dict['num_gpus']
        num_workers = config_dict['num_workers']
        train_iter = config_dict['train_iter']
        plt_dir = config_dict['plt_dir']
        local_dir = config_dict['local_dir']
        local_dir = os.path.join(local_dir, env_name, model_name)
        project_name = local_dir.replace('/', '_')

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

    # init env
    if env_name == 'pensimenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
            "dense_reward": dense_reward,
        }
    elif env_name == 'beerfmtenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
            "dense_reward": dense_reward,
        }
    elif env_name == 'atropineenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
        }
    elif env_name == 'reactorenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
            "compute_diffs_on_reward": compute_diffs_on_reward,
            "dense_reward": dense_reward,
        }
    elif env_name == 'mabupstreamenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
            "dense_reward": dense_reward,
        }
    else:
        raise ValueError('env_name not recognized')
    env = env_creator(env_config)
    
    rl_model = OnlineRLModel(model_name=model_name, ckpt_path=best_checkpoint_path, env="my_env", env_config=env_config, num_gpus=1)


    from smpl.envs.beerfmtenv import BeerFMTEnvGym, MAX_STEPS
    from smpl.envs.utils import normalize_spaces
    # initialize the default policy
    profile_industrial = [11 + 1 / 8 * i for i in range(25)] \
                            + [14] * 95 \
                            + [14 + 2 / 25 * i for i in range(25)] \
                            + [16] * 25 + [16 - 8 / 15 * i for i in range(14)] + [9]*16 # a simple policy
                            

    env = BeerFMTEnvGym(normalize=False)
    state = env.reset(initial_state=[0, 2, 2, 130, 0, 0, 0, 0])
    total_reward = 0.0
    actions = []
    for step in tqdm(range(MAX_STEPS)):
        action = rl_model.predict(state)
        actions.append(action)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    res_rl = info["res_forplot"]
    actions = np.array(actions)
    print("your end at step", step, "and total reward is:", total_reward)
    #your end at step 145 and total reward is: 55.0
    
    profile_inuse = profile_industrial
    env = BeerFMTEnvGym(normalize=False)
    state = env.reset(initial_state=[0, 2, 2, 130, 0, 0, 0, 0])
    total_reward = 0.0
    for step in tqdm(range(MAX_STEPS)):
        action = np.array([profile_inuse[step]], dtype=np.float32)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    res_industrial = info["res_forplot"]
    print("your end at step", step, "and total reward is:", total_reward)
    
    # plots
    plt.figure(dpi=1200)
    plt.subplot(2, 2, 1)
    plt.plot(profile_industrial, label='Industrial', color='blue')
    plt.plot(actions, label='Current_Model', color='blue', linestyle='dashed')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.ylim((0, 18))
    plt.xlabel('Time [h]')
    plt.ylabel('Temperature [\u00B0C]')
    plt.title("Fermentation Profile")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(res_industrial[:, 3], label='Sugar', color='m')
    plt.plot(res_rl[:, 3], color='m', linestyle='dashed')
    plt.plot(res_industrial[:, 4], label='Ethanol', color='orange')
    plt.plot(res_rl[:, 4], color='orange', linestyle='dashed')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.ylim((0, 140))
    plt.xlabel('Time [h]')
    plt.ylabel('Concentration [g/L]')
    plt.title("Sugar and Ethanol Production")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(res_industrial[:, 0], label='Active', color='green')
    plt.plot(res_rl[:, 0], color='green', linestyle='dashed')
    plt.plot(res_industrial[:, 1], label='Lag', color='c')
    plt.plot(res_rl[:, 1], color='c', linestyle='dashed')
    plt.plot(res_industrial[:, 2], label='Dead', color='red')
    plt.plot(res_rl[:, 2], color='red', linestyle='dashed')
    plt.plot(res_industrial[:, 0] + res_industrial[:, 1] + res_industrial[:, 2], label='Total', color='black')
    plt.plot(res_rl[:, 0] + res_rl[:, 1] + res_rl[:, 2], color='black', linestyle='dashed')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.ylim((0, 9))
    plt.xlabel('Time [h]')
    plt.ylabel('Concentration [g/L]')
    plt.title("Biomass Evolution")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(res_industrial[:, 5], label='Diacetyl', color='darkgoldenrod')
    plt.plot(res_rl[:, 5], color='darkgoldenrod', linestyle='dashed')
    plt.plot(res_industrial[:, 6], label='Ethyl Acelate', color='grey')
    plt.plot(res_rl[:, 6], color='grey', linestyle='dashed')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.ylim((0, 1.6))
    plt.xlabel('Time [h]')
    plt.ylabel('Concentration [ppm]')
    plt.title("By-Product Production")
    plt.legend()


    plt.tight_layout()
    plt.savefig('online_rl_infernced.png')