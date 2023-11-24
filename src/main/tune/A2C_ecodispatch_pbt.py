import os
import warnings

import numpy as np
from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import EcoDispatchEnv
import ray
from ray.tune import register_env, Tuner, TuneConfig

from src.metric.metric import OPFMetrics

def make_network_layouts() -> list:
    return [(256, 256, 256), (256, 256, 512), (256, 512, 256), (256, 512, 512), (512, 256, 256), (512, 256, 512), (512, 512, 256), (512, 512, 512)]


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(address="auto", log_to_driver=False, _redis_password=os.environ["redis_password"], include_dashboard=False, dashboard_host="0.0.0.0")

    env_name = "EcoDispatchEnv-v0"

    register_env(env_name, lambda c: EcoDispatchEnv(**c))

    config = A2CConfig()
    config = config.training(use_critic=True,
                             use_gae=False,
                             lr=tune.uniform(5e-5, 4e-4),
                             vf_loss_coeff=tune.uniform(0.5, 1.0),
                             entropy_coeff=tune.uniform(0.0, 0.01),
                             train_batch_size=tune.choice([2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]),
                             model={"fcnet_hiddens": tune.choice(make_network_layouts()), "fcnet_activation": "tanh"},
                             _enable_learner_api=False)

    config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             enable_connectors=False,
                             num_rollout_workers=4,
                             rollout_fragment_length="auto",
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None,
                             create_env_on_local_worker=True)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.choice(list(range(101, 200))), log_sys_usage=False)

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=100)

    config = config.evaluation(evaluation_interval=1200,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

    config = config.callbacks(OPFMetrics)



    hyperparameters_mutations = {
        "lr": tune.uniform(5e-5, 4e-4),
        "vf_loss_coeff": tune.uniform(0.5, 1.0),
        "entropy_coeff": tune.uniform(0.0, 0.01),
        "train_batch_size": [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10],
    }

    scheduler = PopulationBasedTraining(time_attr="training_iteration",
                                        metric="episode_reward_mean",
                                        mode="max",
                                        hyperparam_mutations=hyperparameters_mutations,
                                        perturbation_interval=48,
                                        require_attrs=False)

    checkpoint_config = CheckpointConfig(num_to_keep=None, checkpoint_frequency=48, checkpoint_at_end=True)

    failure_config = FailureConfig(max_failures=3)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=1200), checkpoint_config=checkpoint_config, failure_config=failure_config)

    tune_config = TuneConfig(num_samples=100, reuse_actors=False, scheduler=scheduler)

    results = Tuner("A2C", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    print("-------------------------------------------------------------------------------------------------------")
    best_result_episode = results.get_best_result(metric="evaluation/sampler_results/episode_reward_mean", mode="max", scope="last")
    print('Best result path:', best_result_episode.path)
    for i, j in best_result_episode.config.items():
        print(i, j)

    print("-------------------------------------------------------------------------------------------------------")

    best_result_episode = results.get_best_result(metric="evaluation/sampler_results/custom_metrics/valids_mean", mode="max", scope="last")
    print('Best result path:', best_result_episode.path)
    for i, j in best_result_episode.config.items():
        print(i, j)
    print("-------------------------------------------------------------------------------------------------------")

    ray.shutdown()
