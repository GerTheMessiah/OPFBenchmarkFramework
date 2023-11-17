#! /bin/bash

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_actor_lr qmarket episode_reward_mean optimization actor_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_actor_lr qmarket custom_metrics/valids_mean optimization actor_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_critic_lr qmarket episode_reward_mean optimization critic_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_critic_lr qmarket custom_metrics/valids_mean optimization critic_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_entropy_lr qmarket episode_reward_mean optimization entropy_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_entropy_lr qmarket custom_metrics/valids_mean optimization entropy_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_batch_size qmarket episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_batch_size qmarket custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_actor_network qmarket episode_reward_mean policy_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_actor_network qmarket custom_metrics/valids_mean policy_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_critic_network qmarket episode_reward_mean q_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_critic_network qmarket custom_metrics/valids_mean q_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_tau qmarket episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_tau qmarket custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_reward_scaling qmarket episode_reward_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/SAC/qmarket/SAC_reward_scaling qmarket custom_metrics/valids_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_actor_lr ecodispatch episode_reward_mean optimization actor_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_actor_lr ecodispatch custom_metrics/valids_mean optimization actor_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_critic_lr ecodispatch episode_reward_mean optimization critic_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_critic_lr ecodispatch custom_metrics/valids_mean optimization critic_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_entropy_lr ecodispatch episode_reward_mean optimization entropy_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_entropy_lr ecodispatch custom_metrics/valids_mean optimization entropy_learning_rate

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_batch_size ecodispatch episode_reward_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_batch_size ecodispatch custom_metrics/valids_mean train_batch_size

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_actor_network ecodispatch episode_reward_mean policy_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_actor_network ecodispatch custom_metrics/valids_mean policy_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_critic_network ecodispatch episode_reward_mean q_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_critic_network ecodispatch custom_metrics/valids_mean q_model_config fcnet_hiddens

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_tau ecodispatch episode_reward_mean tau

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_tau ecodispatch custom_metrics/valids_mean tau

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_reward_scaling ecodispatch episode_reward_mean env_config reward_scaling

python statistik.py /fs/dss/data/zahl4814/results/SAC/ecodispatch/SAC_reward_scaling ecodispatch custom_metrics/valids_mean env_config reward_scaling