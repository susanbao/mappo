#!/bin/sh
env="MPE"
num_landmarks=2
num_agents=2
algo="rmappo"
exp="check"
seed_max=1
scenario="simple_push"  # simple_speaker_listener # simple_reference # simple_spread # simple_push
s="s6"
e="e3"
cuda=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=${cuda} nohup python train/train_mpe.py --use_valuenorm --use_popart --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 25 --num_env_steps 2500000 --log_interval 10 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "susanbao" --user_name "susanbao" >./results/temp/mp_${s}_${e}.out 2>./results/temp/mp_${s}_${e}.err &
done