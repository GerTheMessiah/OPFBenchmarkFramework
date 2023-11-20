import os
import warnings

from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.schedulers import PopulationBasedTraining

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
                             optimization_config={"actor_learning_rate": tune.uniform(1e-4, 1e-3), "critic_learning_rate": tune.uniform(1e-4, 1e-3), "entropy_learning_rate": tune.uniform(5e-5, 7e-4)},
                             tau=tune.uniform(0.001, 1.0),
                             train_batch_size=tune.choice([128, 256, 512, 1024]),
                             n_step=1,
                             store_buffer_in_checkpoints=False,
                             num_steps_sampled_before_learning_starts=1024,
                             target_network_update_freq=500,
                             _enable_learner_api=False,
                             replay_buffer_config={"_enable_replay_buffer_api": True, "type": "MultiAgentReplayBuffer", "capacity": 2 ** 18, "storage_unit": "timesteps"})

config = config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

config = config.resources(num_gpus=0, num_cpus_per_worker=1)

config = config.rollouts(batch_mode="complete_episodes", num_envs_per_worker=1, num_rollout_workers=2, rollout_fragment_length=4, enable_connectors=False,
                         observation_filter="MeanStdFilter", preprocessor_pref=None, create_env_on_local_worker=False)

config = config.framework(framework="torch")

config = config.environment(env=env_name, env_config={"eval": False, "reward_scaling": 1 / 40000, "add_act_obs": False}, disable_env_checking=True, normalize_actions=True,
                            clip_actions=True)

config = config.debugging(log_level="ERROR", seed=tune.choice(list(range(101, 200))), log_sys_usage=False)

config = config.rl_module(_enable_rl_module_api=False)

config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=100)

config = config.evaluation(evaluation_interval=15000, evaluation_duration=6720,
                           evaluation_config={"explore": False, "env_config": {"eval": True, "reward_scaling": 1 / 40000, "add_act_obs": False}})

config = config.callbacks(OPFMetrics)

checkpoint_config = CheckpointConfig(num_to_keep=None, checkpoint_frequency=750, checkpoint_at_end=True)

hyperparameters_mutations = {"actor_lr": tune.uniform(5e-5, 4.00e-4), "critic_lr": tune.uniform(5e-4, 2.50e-3), "tau": tune.uniform(0.001, 0.01),
    "train_batch_size": [2 ** 8, 2 ** 9, 2 ** 10], "exploration_config": {"stddev": tune.uniform(0.001, 0.01)}, "policy_delay": [1, 2, 3, 4, 5]}

scheduler = PopulationBasedTraining(time_attr="training_iteration", metric="episode_reward_mean", mode="max", hyperparam_mutations=hyperparameters_mutations,
                                    perturbation_interval=750, require_attrs=False)

failure_config = FailureConfig(max_failures=2)

run_config = RunConfig(stop=MaximumIterationStopper(max_iter=15000), checkpoint_config=checkpoint_config, failure_config=failure_config)

tune_config = TuneConfig(num_samples=100, reuse_actors=False, scheduler=scheduler)

results = Tuner("SAC", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

best_result_episode = results.get_best_result(metric="evaluation/sampler_results/episode_reward_mean", mode="max", scope="last")
print('Best result path:', best_result_episode.path)
print("Best final iteration hyperparameter config:\n", best_result_episode.config)
for i, j in best_result_episode.config.items():
    print(i, j)

print()

best_result_episode = results.get_best_result(metric="evaluation/sampler_results/custom_metrics/valids_mean", mode="max", scope="last")
print('Best result path:', best_result_episode.path)
print("Best final iteration hyperparameter config:\n", best_result_episode.config)
for i, j in best_result_episode.config.items():
    print(i, j)

ray.shutdown()