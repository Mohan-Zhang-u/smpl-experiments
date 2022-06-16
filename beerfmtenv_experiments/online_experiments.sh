#!/bin/bash

for env_name in beerfmtenv
do
    for model_name in ppo ddpg a3c sac
    do
        python online_experiments.py --env_name $env_name --model_name $model_name  --normalize False
    done
done

# for env_name in pensimenv beerfmtenv atropineenv reactorenv mabupstreamenv
# do
#     for model_name in ppo pg ars ddpg a2c a3c sac impala
#     do
#         python online_experiments.py --env_name $env_name --model_name $model_name
#     done
# done