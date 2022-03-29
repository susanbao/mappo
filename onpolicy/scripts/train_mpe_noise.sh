#!/bin/sh
env="MPE"
num_landmarks=3
num_agents=3
algo="rmappo"
exp="check"
seed_max=1
scenario="simple_spread"  # simple_speaker_listener # simple_reference
s="s3"
e="e1"
cuda=0
noise=1.0
noiseName=d10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=${cuda} nohup python train/train_mpe.py --use_valuenorm --use_popart --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 25 --num_env_steps 2500000 --log_interval 10 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --noise_std ${noise} --wandb_name "susanbao" --user_name "susanbao" >./temp/mpnt/mpnt_${s}_${e}_${noiseName}.out 2>./temp/mpnt/mpnt_${s}_${e}_${noiseName}.err &
done