import warnings
import os

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ddpg import DDPGConfig

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import QMarketEnv
import ray
from ray.tune import register_env, Tuner, TuneConfig

from src.metric.metric import OPFMetrics

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(address="auto", log_to_driver=False, _redis_password=os.environ["redis_password"], include_dashboard=False, dashboard_host="0.0.0.0")

    env_name = "QMarketEnv-v0"
    register_env(env_name, lambda c: QMarketEnv(**c))

    config = DDPGConfig()
    config = config.training(twin_q=False,
                             smooth_target_policy=False,
                             target_noise=0.1,
                             critic_lr=0.0016824,
                             actor_lr=4.470888e-5,
                             actor_hiddens=[512, 256, 256],
                             actor_hidden_activation="tanh",
                             critic_hiddens=[256, 512, 512],
                             critic_hidden_activation="tanh",
                             gamma=0.99,
                             tau=0.011193,
                             n_step=1,
                             l2_reg=1e-6,
                             train_batch_size=256,
                             use_huber=False,
                             huber_threshold=1.0,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 18, "storage_unit": "timesteps"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "GaussianNoise", "stddev": 0.0125383, "initial_scale": 1.0, "final_scale": 1.0})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             num_rollout_workers=8,
                             rollout_fragment_length=1,
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 50, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.grid_search([243, 270, 417, 489, 586, 697, 728, 801, 839, 908]))

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0)

    config = config.evaluation(evaluation_interval=20000,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 50, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_frequency=0, checkpoint_at_end=True)

    run_config = RunConfig(verbose=1, stop=MaximumIterationStopper(max_iter=20000), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    res = Tuner("DDPG", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()
