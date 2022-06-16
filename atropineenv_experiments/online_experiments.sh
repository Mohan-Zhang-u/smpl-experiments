#!/bin/bash

for env_name in atropineenv
do
    for model_name in ppo a3c sac ddpg
    do
        python online_experiments.py --env_name $env_name --model_name $model_name
    done
done

# for env_name in pensimenv beerfmtenv atropineenv reactorenv mabupstreamenv
# do
#     for model_name in ppo pg ars ddpg a2c a3c sac impala
#     do
#         python online_experiments.py --env_name $env_name --model_name $model_name
#     done
# done