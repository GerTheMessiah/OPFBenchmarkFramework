import warnings
import os

from ray.rllib.algorithms.td3 import TD3Config

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune.stopper import MaximumIterationStopper

from mlopf.envs.thesis_envs import EcoDispatchEnv
import ray
from ray.tune import register_env, Tuner, TuneConfig


def make_actor_network_layouts():
    return [(256, 256), (256, 512), (512, 256), (512, 512), (256, 256, 256), (256, 256, 512), (256, 512, 256), (256, 512, 512), (512, 256, 256), (512, 256, 512), (512, 512, 256), (512, 512, 512)]

def make_critic_network_layouts():
    return [(256, 256, 256), (256, 256, 512), (256, 512, 256), (256, 512, 512), (512, 256, 256), (512, 256, 512), (512, 512, 256), (512, 512, 512)]


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(address="auto", log_to_driver=False, _redis_password=os.environ["redis_password"], include_dashboard=False, dashboard_host="0.0.0.0")

    env_name = "EcoDispatchEnv-v0"
    register_env(env_name, lambda c: EcoDispatchEnv(**c))

    config = TD3Config()
    config = config.training(twin_q=True,
                             smooth_target_policy=False,
                             actor_lr=tune.uniform(5e-5, 4.00e-4),
                             critic_lr=tune.uniform(5e-4, 2.50e-3),
                             gamma=0.99,
                             tau=tune.uniform(0.001, 0.01),
                             n_step=1,
                             l2_reg=1e-6,
                             train_batch_size=tune.choice([2 ** 8, 2 ** 9, 2 ** 10]),
                             actor_hiddens=tune.choice(make_actor_network_layouts()),
                             actor_hidden_activation="tanh",
                             critic_hiddens=tune.choice(make_critic_network_layouts()),
                             critic_hidden_activation="tanh",
                             _enable_learner_api=False,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 20, "storage_unit": "timesteps"},
                             policy_delay=tune.choice([1, 2, 3, 4, 5])
                             )

    config = config.exploration(explore=True,
                                exploration_config={"type": "GaussianNoise", "stddev": tune.uniform(0.001, 0.05), "initial_scale": 1.0, "final_scale": 1.0})

    config = config.resources(num_gpus=0, num_cpus_per_worker=1)

    config = config.rollouts(batch_mode="complete_episodes",
                             num_envs_per_worker=1,
                             num_rollout_workers=4,
                             rollout_fragment_length=2,
                             enable_connectors=False,
                             observation_filter="MeanStdFilter",
                             preprocessor_pref=None,
                             create_env_on_local_worker=False)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False},
                                disable_env_checking=True,
                                normalize_actions=False,
                                clip_actions=False)

    config = config.debugging(log_level="ERROR", seed=tune.choice(list(range(101, 200))), log_sys_usage=False)

    config = config.rl_module(_enable_rl_module_api=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=1)

    config = config.evaluation(evaluation_interval=110000,
                               evaluation_duration=6720,
                               evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

    checkpoint_config = CheckpointConfig(num_to_keep=None, checkpoint_frequency=10000, checkpoint_at_end=True)

    hyperparameters_mutations = {
        "actor_lr": tune.uniform(5e-5, 4.00e-4),
        "critic_lr": tune.uniform(5e-4, 2.50e-3),
        "tau": tune.uniform(0.001, 0.01),
        "train_batch_size": [2 ** 8, 2 ** 9, 2 ** 10],
        "exploration_config": {"stddev": tune.uniform(0.001, 0.05)},
        "policy_delay": [1, 2, 3, 4, 5]
    }

    scheduler = PopulationBasedTraining(time_attr="training_iteration",
                                        metric="episode_reward_mean",
                                        mode="max",
                                        hyperparam_mutations=hyperparameters_mutations,
                                        perturbation_interval=10000,
                                        burn_in_period=10000,
                                        require_attrs=False)

    failure_config = FailureConfig(max_failures=2)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=110000), checkpoint_config=checkpoint_config, failure_config=failure_config)

    tune_config = TuneConfig(num_samples=50, reuse_actors=False, scheduler=scheduler)

    results = Tuner("TD3", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    best_result_episode = results.get_best_result(metric="evaluation/sampler_results/episode_reward_mean", mode="max", scope="last")

    print("-------------------------------------------------------------------------------------------------------")
    print('Best result path:', best_result_episode.path)
    for i, j in best_result_episode.config.items():
        print(i, j)

    print("-------------------------------------------------------------------------------------------------------")

    ray.shutdown()
