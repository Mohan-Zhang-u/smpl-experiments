#!/bin/bash

for env_name in pensimenv beerfmtenv atropineenv reactorenv mabenv
do
    for model_name in ppo pg ars ddpg a2c a3c sac impala
    do
        python online_experiments.py --env_name $env_name --model_name $model_name --local_dir {$model_name}_ray_results
    done
done
