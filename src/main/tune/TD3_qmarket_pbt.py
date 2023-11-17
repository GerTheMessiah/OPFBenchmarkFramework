import warnings
import os

from ray.rllib.algorithms.td3 import TD3Config

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
import numpy as np
from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import QMarketEnv
import ray
from ray.tune import register_env, Tuner, TuneConfig

from src.metric.metric import OPFMetrics


def make_network_layouts():
    l2 = [tuple([i, j]) for i in [256, 512] for j in [256, 512]]
    l3 = [tuple([i, j, k]) for i in [256, 512] for j in [256, 512] for k in [256, 512]]
    return l2 + l3


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(address="auto", log_to_driver=False, _redis_password=os.environ["redis_password"], include_dashboard=True, dashboard_host="0.0.0.0")

    env_name = "QMarketEnv-v0"
    register_env(env_name, lambda c: QMarketEnv(**c))

    config = TD3Config()
    config = config.training(twin_q=True,
                             smooth_target_policy=False,
                             critic_lr=tune.choice(np.arange(5e-4, 2.60e-3, 1e-4)),
                             actor_lr=tune.choice(np.arange(5e-5, 4.10e-4, 1e-5)),
                             gamma=0.99,
                             tau=tune.choice(np.arange(0.001, 0.011, 0.001)),
                             n_step=1,
                             l2_reg=1e-6,
                             train_batch_size=tune.choice([2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]),
                             actor_hiddens=tune.choice(make_network_layouts()),
                             actor_hidden_activation="tanh",
                             critic_hiddens=tune.choice(make_network_layouts()),
                             critic_hidden_activation="tanh",
                             _enable_learner_api=False,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 18, "storage_unit": "timesteps"},
                             policy_delay=tune.choice([1, 2, 3, 4, 5])
                             )

    config = config.exploration(explore=True,
                                exploration_config={"type": "GaussianNoise", "stddev": tune.choice(np.arange(0.001, 0.051, 0.001)), "initial_scale": 1.0, "final_scale": 1.0})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             enable_connectors=False,
                             num_rollout_workers=2,
                             rollout_fragment_length=4,
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None,
                             create_env_on_local_worker=False)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 50, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.choice(list(range(101, 200))), log_sys_usage=False)

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0)

    config = config.evaluation(evaluation_interval=15000,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 50, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=None, checkpoint_frequency=500, checkpoint_at_end=True)

    hyperparameters_mutations = {
        "critic_lr": np.arange(5e-4, 2.60e-3, 1.0e-4).tolist(),
        "actor_lr": np.arange(5e-5, 4.10e-4, 1e-5).tolist(),
        "tau": np.arange(0.001, 0.01005, 0.0001).tolist(),
        "train_batch_size": [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10],
        "exploration_config": {"stddev": np.arange(0.001, 0.051, 0.001).tolist()},
        "policy_delay": [1, 2, 3, 4, 5]
    }

    scheduler = PopulationBasedTraining(time_attr="training_iteration",
                                        metric="episode_reward_mean",
                                        mode="max",
                                        hyperparam_mutations=hyperparameters_mutations,
                                        perturbation_interval=500,
                                        require_attrs=False)

    failure_config = FailureConfig(max_failures=3)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=15000), checkpoint_config=checkpoint_config, failure_config=failure_config)

    tune_config = TuneConfig(num_samples=100, reuse_actors=False, scheduler=scheduler)

    results = Tuner("TD3", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    best_result = results.get_best_result(metric="episode_reward_mean", mode="max", scope="avg")
    print('Best result path:', best_result.path)
    print("Best final iteration hyperparameter config:\n", best_result.config)

    ray.shutdown()
