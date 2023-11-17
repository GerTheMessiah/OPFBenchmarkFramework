#! /bin/bash

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_actor_lr qmarket episode_reward_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_actor_lr qmarket custom_metrics/valids_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_critic_lr qmarket episode_reward_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_critic_lr qmarket custom_metrics/valids_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_batch_size qmarket episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_batch_size qmarket custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_tau qmarket episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_tau qmarket custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_actor_network qmarket episode_reward_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_actor_network qmarket custom_metrics/valids_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_critic_network qmarket episode_reward_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_critic_network qmarket custom_metrics/valids_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_std_explore qmarket episode_reward_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_std_explore qmarket custom_metrics/valids_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_reward_scaling qmarket episode_reward_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/DDPG/qmarket/DDPG_reward_scaling qmarket custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_actor_lr ecodispatch episode_reward_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_actor_lr ecodispatch custom_metrics/valids_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_critic_lr ecodispatch episode_reward_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_critic_lr ecodispatch custom_metrics/valids_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_batch_size ecodispatch episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_batch_size ecodispatch custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_tau ecodispatch episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_tau ecodispatch custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_actor_network ecodispatch episode_reward_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_actor_network ecodispatch custom_metrics/valids_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_critic_network ecodispatch episode_reward_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_critic_network ecodispatch custom_metrics/valids_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_std_explore ecodispatch episode_reward_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_std_explore ecodispatch custom_metrics/valids_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_reward_scaling ecodispatch episode_reward_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/DDPG/ecodispatch/DDPG_reward_scaling ecodispatch custom_metrics/valids_mean env_config reward_scaling