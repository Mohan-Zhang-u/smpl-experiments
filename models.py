import numpy as np
import torch
import d3rlpy
import ray
import json
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from mzutils import PIDModel
import mpctools as mpc
import yaml
import os
import pickle
from typing import Union, List, Tuple, Any


def set_env_config(env_name, normalize=None, dense_reward=None, reward_on_steady=None, reward_on_absolute_efactor=None,
                   compute_diffs_on_reward=None,
                   standard_reward_style=None, initial_state_deviation_ratio=None):
    if env_name == 'pensimenv':
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    elif env_name == 'beerfmtenv':
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    elif env_name == 'atropineenv':
        assert normalize is not None
        assert dense_reward is not None
        assert reward_on_steady is not None
        assert reward_on_absolute_efactor is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "reward_on_steady": False,
            "reward_on_absolute_efactor": True,
        }
    elif env_name == 'reactorenv':
        assert normalize is not None
        assert dense_reward is not None
        assert compute_diffs_on_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
            "compute_diffs_on_reward": compute_diffs_on_reward,
        }
    elif env_name == 'mabupstreamenv':
        assert normalize is not None
        assert dense_reward is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
        }
    elif env_name == 'mabenv':
        assert normalize is not None
        assert dense_reward is not None
        assert standard_reward_style is not None
        assert initial_state_deviation_ratio is not None
        env_config = {
            "env_name": env_name,
            "normalize": normalize,
            "dense_reward": dense_reward,
            "standard_reward_style": standard_reward_style,
            "initial_state_deviation_ratio": initial_state_deviation_ratio,
        }
    else:
        raise ValueError('env_name not recognized')
    return env_config


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


class OnlineRLModel(object):
    def __init__(self, model_name, ckpt_path, env, config_dir, num_gpus=1):
        # config_dir is something like 'ray_results/beerfmtenv/ppo/PPOTrainer/PPOTrainer_my_env/params.pkl'
        # ckpt_path is something like 'ray_results/beerfmtenv/ppo/PPOTrainer/PPOTrainer_my_env/checkpoint_002000/checkpoint-2000'
        with open(config_dir, 'rb') as fp:
            config = pickle.load(fp)
        config['num_gpus'] = num_gpus
        # since there are different versions of rllib, there are possibilities that some of the configs need to be removed. Please use the code below:
        # if 'keep_per_episode_custom_metrics' in config:
        #     config.pop('keep_per_episode_custom_metrics')
        # if 'output_config' in config:
        #     config.pop('output_config')
        # if 'disable_env_checking' in config:
        #     config.pop('disable_env_checking')
        if num_gpus == 0:
            config['num_gpus_per_worker'] = 0
        if model_name == 'ppo':
            import ray.rllib.agents.ppo as ppo
            # config = ppo.DEFAULT_CONFIG.copy()
            # config["num_workers"] = 0
            self.agent = ppo.PPOTrainer(config, env=env)
        elif model_name == 'pg':
            import ray.rllib.agents.pg as pg
            # config = pg.DEFAULT_CONFIG.copy()
            # config["num_workers"] = 0
            self.agent = pg.PGTrainer(config, env=env)
        elif model_name == 'ars':
            import ray.rllib.agents.ars as ars
            # config = ars.DEFAULT_CONFIG.copy()
            # config["num_workers"] = 1
            self.agent = ars.ARSTrainer(config, env=env)
        elif model_name == 'ddpg':
            import ray.rllib.agents.ddpg as ddpg
            # config = ddpg.DEFAULT_CONFIG.copy()
            # config["framework"] = "torch"
            # config["env_config"] = set_env_config('beerfmtenv', normalize=False, dense_reward=True)
            # config['num_gpus'] = 0
            # config['num_gpus_per_worker'] = 0
            # config["num_workers"] = 1
            # config["simple_optimizer"] = True
            config['num_gpus'] = 0  # remove gpu allocation for temp ray bug fix.
            self.agent = ddpg.DDPGTrainer(config, env=env)
        elif model_name == 'apex_ddpg':
            import ray.rllib.agents.ddpg.apex as apex
            # config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
            # config["num_workers"] = 1
            # config["simple_optimizer"] = True
            self.agent = apex.DDPGTrainer(config, env=env)
        elif model_name == 'a3c':
            import ray.rllib.agents.a3c as a3c
            self.agent = a3c.A3CTrainer(config, env=env)
            # config = a3c.DEFAULT_CONFIG.copy()
            # config["num_workers"] = 1
        elif model_name == 'sac':
            import ray.rllib.agents.sac as sac
            # config = sac.DEFAULT_CONFIG.copy()
            # config["num_workers"] = 0
            self.agent = sac.SACTrainer(config, env=env)
        elif model_name == 'impala':
            import ray.rllib.agents.impala as impala
            # config = impala.DEFAULT_CONFIG.copy()
            # config["framework"] = "torch"
            # config['num_gpus'] = 0
            # config['num_gpus_per_worker'] = 0
            # config["num_workers"] = 0
            self.agent = impala.ImpalaTrainer(config, env=env)
        elif model_name == 'a2c':
            import ray.rllib.agents.a3c.a2c as a2c
            #   config = a2c.A2C_DEFAULT_CONFIG.copy()
            #   config["num_workers"] = 0
            self.agent = a2c.A2CTrainer(config, env=env)
        else:
            raise ValueError(f'{model_name} not recognized')
        self.agent.restore(ckpt_path)

    def predict(self, state):
        return self.agent.compute_action(state)


class OfflineRLModel(object):
    def __init__(self, algo_name, config_dict_loc='offline_experiments.yaml'):
        with open(config_dict_loc, 'r') as fp:
            config_dict = yaml.safe_load(fp)
        best_loc = config_dict['best_loc']
        best_params = os.path.join(best_loc, f'{algo_name}/params.json')
        best_ckpt = os.path.join(best_loc, f'{algo_name}/best.pt')

        if algo_name == 'CQL':
            curr_algo = d3rlpy.algos.CQL.from_json(best_params)
        elif algo_name == 'PLAS':
            curr_algo = d3rlpy.algos.PLAS.from_json(best_params)
        elif algo_name == 'PLASWithPerturbation':
            curr_algo = d3rlpy.algos.PLASWithPerturbation.from_json(best_params)
        elif algo_name == 'DDPG':
            curr_algo = d3rlpy.algos.DDPG.from_json(best_params)
        elif algo_name == 'BC':
            curr_algo = d3rlpy.algos.BC.from_json(best_params)
        elif algo_name == 'TD3':
            curr_algo = d3rlpy.algos.TD3.from_json(best_params)
        elif algo_name == 'BEAR':
            curr_algo = d3rlpy.algos.BEAR.from_json(best_params)
        elif algo_name == 'SAC':
            curr_algo = d3rlpy.algos.SAC.from_json(best_params)
        elif algo_name == 'BCQ':
            curr_algo = d3rlpy.algos.BCQ.from_json(best_params)
        elif algo_name == 'CRR':
            curr_algo = d3rlpy.algos.CRR.from_json(best_params)
        elif algo_name == 'AWR':
            curr_algo = d3rlpy.algos.AWR.from_json(best_params)
        elif algo_name == 'AWAC':
            curr_algo = d3rlpy.algos.AWAC.from_json(best_params)
        elif algo_name == 'COMBO':
            curr_algo = d3rlpy.algos.COMBO.from_json(best_params)
        elif algo_name == 'MOPO':
            curr_algo = d3rlpy.algos.MOPO.from_json(best_params)
        else:
            raise Exception("algo_name is invalid!", algo_name)

        curr_algo.load_model(best_ckpt)
        self.curr_algo = curr_algo

    def predict(self, state):
        try:
            inp = self.curr_algo.predict(state)  # shape (1,x)
            return inp[0]  # shape (x,)
        except AssertionError:
            inp = self.curr_algo.predict([state])
            return inp[0]


class CustomPIDModel(PIDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, state):
        """
        :param state: control state of the system. The state is a list of length len_c, each of the element cooresponding to an action.
        """
        state = [state[0], state[2]]
        return super().predict(state)
