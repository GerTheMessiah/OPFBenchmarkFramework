#! /bin/bash

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_actor_lr qmarket episode_reward_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_actor_lr qmarket custom_metrics/valids_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_critic_lr qmarket episode_reward_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_critic_lr qmarket custom_metrics/valids_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_batch_size qmarket episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_batch_size qmarket custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_tau qmarket episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_tau qmarket custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_actor_network qmarket episode_reward_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_actor_network qmarket custom_metrics/valids_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_critic_network qmarket episode_reward_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_critic_network qmarket custom_metrics/valids_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_std_explore qmarket episode_reward_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_std_explore qmarket custom_metrics/valids_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/TD3/qmarket/TD3_reward_scaling qmarket custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/work/zahl4814/results/TD3/qmarket/TD3_policy_delay qmarket episode_reward_mean policy_delay

python statistik.py /fs/dss/work/zahl4814/results/TD3/qmarket/TD3_policy_delay qmarket custom_metrics/valids_mean policy_delay

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_actor_lr ecodispatch episode_reward_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_actor_lr ecodispatch custom_metrics/valids_mean actor_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_critic_lr ecodispatch episode_reward_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_critic_lr ecodispatch custom_metrics/valids_mean critic_lr

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_batch_size ecodispatch episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_batch_size ecodispatch custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_tau ecodispatch episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_tau ecodispatch custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_actor_network ecodispatch episode_reward_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_actor_network ecodispatch custom_metrics/valids_mean actor_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_critic_network ecodispatch episode_reward_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_critic_network ecodispatch custom_metrics/valids_mean critic_hiddens

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_std_explore ecodispatch episode_reward_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_std_explore ecodispatch custom_metrics/valids_mean exploration_config stddev

python statistik.py /fs/dss/data/zahl4814/results/TD3/ecodispatch/TD3_reward_scaling ecodispatch custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/work/zahl4814/results/TD3/ecodispatch/TD3_policy_delay ecodispatch episode_reward_mean policy_delay

python statistik.py /fs/dss/work/zahl4814/results/TD3/ecodispatch/TD3_policy_delay ecodispatch custom_metrics/valids_mean policy_delay
