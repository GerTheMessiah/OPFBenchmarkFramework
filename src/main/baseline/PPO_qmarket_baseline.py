import warnings
import os

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.air import RunConfig, CheckpointConfig

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

    config = PPOConfig()
    config = config.training(use_critic=True,
                             use_gae=False,
                             use_kl_loss=False,
                             lr=0.0001,
                             train_batch_size=2 ** 13,
                             sgd_minibatch_size=1024,
                             num_sgd_iter=5,
                             clip_param=0.2,
                             vf_loss_coeff=0.5,
                             entropy_coeff=0.0,
                             shuffle_sequences=True,
                             gamma=0.99,
                             model={"fcnet_hiddens": [256, 256, 256], "fcnet_activation": "tanh"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             num_rollout_workers=11,
                             rollout_fragment_length="auto",
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name, env_config={"eval": False, "reward_scaling": 1 / 50, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR",
                              seed=tune.grid_search([243, 270, 417, 489, 586, 697, 728, 801, 839, 908]),
                              log_sys_usage=False)

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=1)

    config = config.evaluation(evaluation_interval=470,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 50, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_frequency=100, checkpoint_at_end=True)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=1000), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    Tuner("PPO", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()
