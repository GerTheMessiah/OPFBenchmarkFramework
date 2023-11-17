import os
import warnings

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.a2c import A2CConfig

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

    config = A2CConfig()
    config = config.training(use_critic=True,
                             use_gae=False,
                             lr=1e-4,
                             grad_clip=40.0,
                             vf_loss_coeff=1.0,
                             entropy_coeff=tune.grid_search([0.01, 0.005, 0.001, 0.0005, 0.0001]),
                             train_batch_size=1024,
                             model={"fcnet_hiddens": [256, 256, 256], "fcnet_activation": "tanh"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1, num_cpus_per_learner_worker=1)

    config = config.rollouts(batch_mode="complete_episodes", num_envs_per_worker=1, num_rollout_workers=7, rollout_fragment_length="auto", observation_filter="MeanStdFilter",
                             preprocessor_pref=None)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.grid_search([5, 15, 30, 35, 45]))

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=100)

    config = config.evaluation(evaluation_interval=None,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_frequency=0, checkpoint_at_end=False)

    run_config = RunConfig(verbose=1, stop=MaximumIterationStopper(max_iter=600), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    res = Tuner("A2C", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()