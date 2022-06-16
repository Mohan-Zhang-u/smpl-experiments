import yaml
import logging, wandb
import numpy as np
from datetime import datetime
import pandas as pd
import json
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
        from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData, NUM_STEPS
        recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
                    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                    FG: Recipe(FG_DEFAULT_PROFILE, FG),
                    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}
        recipe_combo = RecipeCombo(recipe_dict=recipe_dict)
        # set up the environment
        env = PenSimEnvGym(recipe_combo=recipe_combo, normalize=env_config["normalize"], dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'beerfmtenv':
        from smpl.envs.beerfmtenv import BeerFMTEnvGym
        env = BeerFMTEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'atropineenv':
        from smpl.envs.atropineenv import AtropineEnvGym
        env = AtropineEnvGym(normalize=env_config["normalize"]) # by default uses reward on steady.
    elif env_config["env_name"] == 'reactorenv':
        from smpl.envs.reactorenv import ReactorEnvGym
        env = ReactorEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"], compute_diffs_on_reward=env_config["compute_diffs_on_reward"])
    elif env_config["env_name"] == 'mabupstreamenv':
        from smpl.envs.mabenv import MAbUpstreamEnvGym
        env = MAbUpstreamEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"])
    elif env_config["env_name"] == 'mabenv':
        from smpl.envs.mabenv import MAbEnvGym
        env = MAbEnvGym(normalize=env_config["normalize"], dense_reward=env_config["dense_reward"], standard_reward_style=env_config["standard_reward_style"], initial_state_deviation_ratio=env_config["initial_state_deviation_ratio"]) 
    else:
        raise ValueError('env_name not recognized')
    return env
register_env("my_env", env_creator)

if __name__ == "__main__":
    with open(f'online_experiments.yaml', 'r') as fp:
        config_dict = yaml.safe_load(fp)
        env_name = config_dict['env_name']
        model_name = config_dict['model_name']
        normalize = config_dict['normalize']
        dense_reward = config_dict['dense_reward']
        scheduler_name = config_dict['scheduler_name']
        time_budget_s = config_dict['time_budget_s']
        standard_reward_style = config_dict.get('standard_reward_style', None)
        
        best_checkpoint_path = config_dict.get('best_checkpoint_path', None)
        use_tune = config_dict['use_tune']
        num_gpus = config_dict['num_gpus']
        num_workers = config_dict['num_workers']
        # num_gpus_per_worker = config_dict['num_gpus_per_worker']
        train_iter = config_dict['train_iter']
        plt_dir = config_dict['plt_dir']
        local_dir = config_dict['local_dir']

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
    elif env_name == 'mabenv':
        env_config = {
            "env_name": env_name,
            "normalize": normalize, 
            "dense_reward": dense_reward,
            "standard_reward_style": standard_reward_style,
        }
    else:
        raise ValueError('env_name not recognized')
    
    # init algorithm
    if model_name == 'ppo':
        import ray.rllib.agents.ppo as ppo
        imported_algo = ppo
        rl_trainer = ppo.PPOTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
        # grid_search for parameter tunning
        # turns out default is the best.
        # config["lambda"] = tune.grid_search([1.0, 0.9, 0.8, 0.7])
        # config["kl_coeff"] = tune.grid_search([0.4, 0.3, 0.2, 0.1])
        # config["rollout_fragment_length"] = tune.grid_search([400, 300, 200])
        # config["entropy_coeff"] = tune.grid_search([0.0, 0.1, 0.2])
        # config["kl_target"] = tune.grid_search([0.01, 0.02, 0.03])
    elif model_name == 'pg':
        import ray.rllib.agents.pg as pg
        imported_algo = pg
        rl_trainer = pg.PGTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
    elif model_name == 'ars':
        import ray.rllib.agents.ars as ars
        imported_algo = ars
        rl_trainer = ars.ARSTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
    elif model_name == 'ddpg':
        import ray.rllib.agents.ddpg as ddpg
        imported_algo = ddpg
        rl_trainer = ddpg.DDPGTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
    elif model_name == 'apex_ddpg':
        import ray.rllib.agents.ddpg.apex as apex
        imported_algo = apex
        rl_trainer = apex.DDPGTrainer
        config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    elif model_name == 'a3c':
        import ray.rllib.agents.a3c as a3c
        imported_algo = a3c
        rl_trainer = a3c.A3CTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
        config["lr"] = 0.00005
    elif model_name == 'sac':
        import ray.rllib.agents.sac as sac
        imported_algo = sac
        rl_trainer = sac.SACTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
    elif model_name == 'impala':
        import ray.rllib.agents.impala as impala
        imported_algo = impala
        rl_trainer = impala.ImpalaTrainer
        config = imported_algo.DEFAULT_CONFIG.copy()
    elif model_name == 'a2c':
        import ray.rllib.agents.a3c.a2c as a2c
        imported_algo = a2c
        rl_trainer = a2c.A2CTrainer
        config = imported_algo.A2C_DEFAULT_CONFIG.copy()
    else:
        raise ValueError('model_name not recognized')
    
    if use_tune:
        with wandb.init(project = project_name, sync_tensorboard = True) as run:
            config["env"] = "my_env"
            config["num_gpus"] = num_gpus
            config["framework"] = "torch"
            config["num_workers"] = num_workers
            config["evaluation_num_workers"] = 1
            # config["num_gpus_per_worker"] = num_gpus_per_worker
            config["evaluation_interval"] = int(train_iter / 10)
            config["evaluation_duration"] = 10
            config["env_config"] = env_config
            config["logger_config"] = {
                "wandb": {
                    "project": project_name,
                    "log_config": True, 
                }
            }
            if scheduler_name == 'asha_scheduler':
                scheduler = tune.schedulers.ASHAScheduler(
                    time_attr='training_iteration',
                    metric='episode_reward_mean',
                    mode='max',
                    max_t=train_iter,
                    grace_period=10,
                    reduction_factor=3,
                    brackets=1
                )
                analysis = tune.run(rl_trainer, 
                    metric='episode_reward_mean',
                    mode='max',
                    time_budget_s=time_budget_s,
                    config=config, 
                    local_dir=local_dir,
                    checkpoint_freq=1,
                    checkpoint_at_end=True,
                    keep_checkpoints_num=5,
                    checkpoint_score_attr="episode_reward_mean", # Specifies by which attribute to rank the best checkpoint. Default is increasing order. If attribute starts with min- it will rank attribute in decreasing order, i.e. min-validation_loss.
                    stop={"training_iteration": train_iter},
                    scheduler=scheduler,
                    loggers=DEFAULT_LOGGERS + (WandbLogger, )
                )
            elif scheduler_name == 'fifo_scheduler':
                analysis = tune.run(rl_trainer, 
                    metric='episode_reward_mean',
                    mode='max',
                    time_budget_s=time_budget_s,
                    config=config, 
                    local_dir=local_dir,
                    checkpoint_freq=1,
                    checkpoint_at_end=True,
                    keep_checkpoints_num=5,
                    checkpoint_score_attr="episode_reward_mean", # Specifies by which attribute to rank the best checkpoint. Default is increasing order. If attribute starts with min- it will rank attribute in decreasing order, i.e. min-validation_loss.
                    stop={"training_iteration": train_iter},
                    loggers=DEFAULT_LOGGERS + (WandbLogger, )
                )
            else:
                raise NotImplementedError
            
            print("Best config: ", analysis.get_best_config(
                metric="episode_reward_mean", mode="max"
            ))
        
    else:
        CHECKPOINT_ROOT = local_dir + "/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        best_episode_reward_mean = -1e8
        checkpoint_file = ""
        info = ray.init(ignore_reinit_error=True, num_gpus=1)
        status = "reward {:6.2f} {:6.2f} {:6.2f}  len {:4.2f}  saved {}"
        config = imported_algo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = num_gpus
        config["env_config"] = env_config
        config["framework"] = "torch"
        config["num_workers"] = num_workers
        config["evaluation_num_workers"] = 1
        config["evaluation_interval"] = int(train_iter / 10)
        config["evaluation_duration"] = 10
        agent = rl_trainer(config, env="my_env")
        df = pd.DataFrame(columns=[ "min_reward", "avg_reward", "max_reward", "steps", "checkpoint"])
        for i in range(train_iter):
            result = agent.train()
            checkpoint_file = agent.save(CHECKPOINT_ROOT)
            if result["episode_reward_mean"] > best_episode_reward_mean:
                best_episode_reward_mean = result["episode_reward_mean"]
                best_iter = i
                best_checkpoint_file = checkpoint_file
            row = [
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                checkpoint_file,
            ]
            df.loc[len(df)] = row
            print(status.format(*row))
        df.to_csv(f"{CHECKPOINT_ROOT}/{model_name}_results.csv")
        result_dict = {"best_episode_reward_mean": best_episode_reward_mean, "best_iter": best_iter, "checkpoint_file": best_checkpoint_file}
        json.dump(result_dict, open(f"{CHECKPOINT_ROOT}/{model_name}_result.json", 'w+'))     
        # agent.restore(checkpoint_file)
        # state=env.reset()
        # a=agent.compute_action(state)
        # print(a)
