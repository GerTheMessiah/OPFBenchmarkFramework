#! /bin/bash

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_lr qmarket episode_reward_mean lr

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_lr qmarket custom_metrics/valids_mean lr

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_batch_size qmarket episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_batch_size qmarket custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_network qmarket episode_reward_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_network qmarket custom_metrics/valids_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_value_loss qmarket episode_reward_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_value_loss qmarket custom_metrics/valids_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_entropy qmarket episode_reward_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_entropy qmarket custom_metrics/valids_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_grad_clip qmarket episode_reward_mean grad_clip

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_grad_clip qmarket custom_metrics/valids_mean grad_clip

python statistik.py /fs/dss/data/zahl4814/results/A2C/qmarket/A2C_reward_scaling qmarket custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_lr ecodispatch episode_reward_mean lr

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_lr ecodispatch custom_metrics/valids_mean lr

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_batch_size ecodispatch episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_batch_size ecodispatch custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_network ecodispatch episode_reward_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_network ecodispatch custom_metrics/valids_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_value_loss ecodispatch episode_reward_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_value_loss ecodispatch custom_metrics/valids_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_entropy ecodispatch episode_reward_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_entropy ecodispatch custom_metrics/valids_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_grad_clip ecodispatch episode_reward_mean grad_clip

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_grad_clip ecodispatch custom_metrics/valids_mean grad_clip

python statistik.py /fs/dss/data/zahl4814/results/A2C/ecodispatch/A2C_reward_scaling ecodispatch custom_metrics/valids_mean env_config reward_scaling