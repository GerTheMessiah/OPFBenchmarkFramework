import warnings
import os

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import QMarketEnv, EcoDispatchEnv
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

    config = PPOConfig()
    config = config.training(use_critic=True,
                             use_gae=False,
                             use_kl_loss=False,
                             lr=tune.choice(np.arange(7e-5, 7e-4, 1e-5).tolist()),
                             train_batch_size=tune.choice([2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]),
                             sgd_minibatch_size=tune.choice([128, 256, 512, 1024]),
                             num_sgd_iter=tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
                             clip_param=tune.choice(np.arange(0.1, 0.3, 0.01).tolist()),
                             vf_loss_coeff=tune.choice(np.arange(0.5, 1.0, 0.01).tolist()),
                             entropy_coeff=tune.choice(np.arange(0.0, 0.02, 0.0001).tolist()),
                             shuffle_sequences=True,
                             gamma=0.99,
                             model={"fcnet_hiddens": tune.choice(make_network_layouts()), "fcnet_activation": "tanh"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1, num_cpus_per_learner_worker=1)

    config = config.rollouts(batch_mode="complete_episodes", num_envs_per_worker=1, num_rollout_workers=4, rollout_fragment_length="auto", observation_filter="MeanStdFilter",
                             preprocessor_pref=None)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name, env_config={"eval": False, "reward_scaling": 1 / 50, "add_act_obs": False}, disable_env_checking=True, normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.choice(list(range(101, 200))), log_sys_usage=False)

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0)

    config = config.evaluation(evaluation_interval=250, evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 50, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)

    checkpoint_config = CheckpointConfig(num_to_keep=None, checkpoint_frequency=10, checkpoint_at_end=True)

    hyperparameters_mutations = {"lr": np.arange(7.5e-5, 7.5e-4, 0.1e-5).tolist(), "train_batch_size": [2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13],
        "sgd_minibatch_size": [128, 256, 512, 1024], "num_sgd_iter": [3, 4, 5, 6, 7, 8, 9, 10], "clip_param": np.arange(0.1, 0.3, 0.01).tolist(),
        "vf_loss_coeff": np.arange(0.5, 1.0, 0.01).tolist(), "entropy_coeff": np.arange(0.0, 0.02, 0.0001).tolist(), }

    scheduler = PopulationBasedTraining(time_attr="training_iteration", metric="episode_reward_mean", mode="max", hyperparam_mutations=hyperparameters_mutations,
                                        perturbation_interval=10, require_attrs=False)

    failure_config = FailureConfig(max_failures=3)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=250), checkpoint_config=checkpoint_config, failure_config=failure_config)

    tune_config = TuneConfig(num_samples=100, reuse_actors=False, scheduler=scheduler)

    results = Tuner("PPO", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    best_result = results.get_best_result(metric="episode_reward_mean", mode="max", scope="avg")

    print('Best result path:', best_result.path)
    print("Best final iteration hyperparameter config:\n", best_result.config)

    ray.shutdown()
