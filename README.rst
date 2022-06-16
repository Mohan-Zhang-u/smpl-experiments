.. _SMPL: https://github.com/smpl-env/smpl

This repo hosts the code for the online and offline reinforcement learning experiments on The Simulated Industrial Manufacturing and Process Control Learning Environments (`SMPL`_).

Install Requirements
====================
.. code-block::

    $ pip install -r requirements.txt



Online Experiments
==================

Online Training
---------------

You simply need to run the :code:`online_experiments.sh` script. Note that you may want to edit the :code:`online_experiments.yaml` for different configurations.
Or, optionally, you can go into the specific directory of environments (e.g. mabenv_experiments) and execute the :code:`online_experiments.sh` (which executes :code:`online_experiments.py` for the online RL algorithms) there for their specific configurations. Moreover, you can edit the configurations in :code:`online_experiments.yaml`. 

Online Inference
----------------

After you trained an online RL algorithm, you could do the inference with :code:`online_inference.py`. You need to set the env_name, model_names, best_checkpoint_paths and config_dirs accordingly such that the correct checkpoint(s) are loaded. You can also set the plot configurations to visualize how the trained algorithm actually performs. For more details, please consult the docstring in :code:`online_inference.py` and `this documentation <https://smpl-env.readthedocs.io/en/latest/index.html>`_.

Offline Experiments
===================

Offline Training
----------------

You first need to generate a dataset with the baseline algorithm using the script :code:`offline_data_generation.py` located in :code:`{env_name}_experiments`. After successfully generated the training, evaluating and testing initial states and datasets, you can then use the :code:`offlineRL_training.py` to train the offline RL algorithms. Don't forget that you can edit the configurations in :code:`offline_experiments.yaml`. 

Offline Inference
-----------------

The :code:`OFFLINE_BEST.yaml` in {env_name}_experiments specifies the location of your current offline RL experiments. For example, if you finished the experiment of Behavior Cloning and you put :code:`"d3rlpy_logs/42"` in the :code:`OFFLINE_BEST.yaml`, then you should be able to locate the best checkpoint in :code:`d3rlpy_logs/42/BC/best.pt`, which is the checkpoint used to perform the inference with :code:`offline_inference.py`. Again you can set the plot configurations to analyze and visualize the results.