#! /bin/bash

python statistik.py /fs/dss/work/zahl4814/results/PPO/qmarket/PPO_lr qmarket episode_reward_mean lr

python statistik.py /fs/dss/work/zahl4814/results/PPO/qmarket/PPO_lr qmarket custom_metrics/valids_mean lr

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_batch_size qmarket episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_batch_size qmarket custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_mini_batch_size qmarket episode_reward_mean sgd_minibatch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_mini_batch_size qmarket custom_metrics/valids_mean sgd_minibatch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_num_iter qmarket episode_reward_mean num_sgd_iter

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_num_iter qmarket custom_metrics/valids_mean num_sgd_iter

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_clip qmarket episode_reward_mean clip_param

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_clip qmarket custom_metrics/valids_mean clip_param

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_value_loss qmarket episode_reward_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_value_loss qmarket custom_metrics/valids_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_entropy qmarket episode_reward_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_entropy qmarket custom_metrics/valids_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_network qmarket episode_reward_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_network qmarket custom_metrics/valids_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/PPO/qmarket/PPO_reward_scaling qmarket custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_lr ecodispatch episode_reward_mean lr

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_lr ecodispatch custom_metrics/valids_mean lr

python statistik.py /fs/dss/work/zahl4814/results/PPO/ecodispatch/PPO_batch_size ecodispatch episode_reward_mean train_batch_size

python statistik.py /fs/dss/work/zahl4814/results/PPO/ecodispatch/PPO_batch_size ecodispatch custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_mini_batch_size ecodispatch episode_reward_mean sgd_minibatch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_mini_batch_size ecodispatch custom_metrics/valids_mean sgd_minibatch_size

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_num_iter ecodispatch episode_reward_mean num_sgd_iter

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_num_iter ecodispatch custom_metrics/valids_mean num_sgd_iter

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_clip ecodispatch episode_reward_mean clip_param

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_clip ecodispatch custom_metrics/valids_mean clip_param

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_value_loss ecodispatch episode_reward_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_value_loss ecodispatch custom_metrics/valids_mean vf_loss_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_entropy ecodispatch episode_reward_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_entropy ecodispatch custom_metrics/valids_mean entropy_coeff

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_network ecodispatch episode_reward_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_network ecodispatch custom_metrics/valids_mean model fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/PPO/ecodispatch/PPO_reward_scaling ecodispatch custom_metrics/valids_mean env_config reward_scaling