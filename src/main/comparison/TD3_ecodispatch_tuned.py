import warnings
import os

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.td3 import TD3Config

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import EcoDispatchEnv
import ray
from ray.tune import register_env, Tuner, TuneConfig

from src.metric.metric import OPFMetrics

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(address="auto", log_to_driver=False, _redis_password=os.environ["redis_password"], include_dashboard=False, dashboard_host="0.0.0.0")

    env_name = "EcoDispatchEnv-v0"
    register_env(env_name, lambda c: EcoDispatchEnv(**c))

    config = TD3Config()
    config = config.training(twin_q=True,
                             smooth_target_policy=False,
                             actor_lr=6.740772e-5,
                             critic_lr=0.000834612,
                             actor_hiddens=[512, 512, 512],
                             actor_hidden_activation="tanh",
                             critic_hiddens=[512, 512, 512],
                             critic_hidden_activation="tanh",
                             gamma=0.99,
                             tau=0.00385487,
                             n_step=1,
                             train_batch_size=1024,
                             policy_delay=1,
                             use_huber=False,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 20, "storage_unit": "timesteps"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "GaussianNoise", "stddev": 0.0021306, "initial_scale": 1.0, "final_scale": 1.0})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1, num_cpus_per_learner_worker=1, num_learner_workers=2)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             num_rollout_workers=7,
                             rollout_fragment_length=1,
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None,
                             create_env_on_local_worker=False,
                             enable_connectors=False)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.grid_search([243, 270, 417, 489, 586, 697, 728, 801, 839, 908]))

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=1)

    config = config.evaluation(evaluation_interval=100000,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_frequency=10000, checkpoint_at_end=True)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=100000), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    res = Tuner("TD3", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()