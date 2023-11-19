import os
import warnings

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.sac import SACConfig

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

    config = SACConfig()
    config = config.training(twin_q=True,
                             q_model_config={"fcnet_hiddens": [256, 256, 256], "fcnet_activation": "tanh"},
                             policy_model_config={"fcnet_hiddens": [256, 256, 256], "fcnet_activation": "tanh"},
                             optimization_config={"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "entropy_learning_rate": 3e-4},
                             tau=0.005,
                             train_batch_size=tune.grid_search([128, 256, 512, 1024]),
                             n_step=1,
                             store_buffer_in_checkpoints=False,
                             num_steps_sampled_before_learning_starts=1024,
                             target_network_update_freq=1,
                             _enable_learner_api=False,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 18, "storage_unit": "timesteps"})

    config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1, num_cpus_per_learner_worker=1)

    config = config.rollouts(batch_mode="complete_episodes", num_envs_per_worker=1, num_rollout_workers=4, rollout_fragment_length=2, observation_filter="MeanStdFilter",
                             preprocessor_pref=None)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=True,
                                clip_actions=True)

    config = config.debugging(log_level="ERROR", seed=tune.grid_search([5, 15, 30, 35, 45]))

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=100)

    config = config.evaluation(evaluation_interval=None,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_frequency=0, checkpoint_at_end=True)

    run_config = RunConfig(verbose=1, stop=MaximumIterationStopper(max_iter=15000), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    res = Tuner("SAC", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()