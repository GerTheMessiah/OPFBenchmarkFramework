import os
import json
import sys
import numpy
from statistics import mean, stdev

import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':

    env = sys.argv[1]  # qmarket | ecodispatch

    policy = sys.argv[2]

    top_dir_tuned = f"/fs/dss/work/zahl4814/comparison_new/{env}"
    top_dir_baseline = f"/fs/dss/work/zahl4814/baseline_new/{env}"

    fig = plt.figure(figsize=(10, 7.5))
    plt.xlabel('Training iterations')
    plt.ylabel('Episode reward mean')

    # plt.xlabel('Steps sampled')

    if env == "qmarket":
        plt.ylim(-0.11, -0.02)
    else:
        plt.ylim(-1.4, -0.4)

    ax = fig.gca()

    if env == 'qmarket':
        ax.set_yticks(numpy.arange(-0.11, -0.02, 0.01))
    else:
        ax.set_yticks(numpy.arange(-1.4, -0.4, 0.1))

    colors_tuned = ["#8D1B1B", "#692792", "#02BCD3", "#F47F17", "#80B344"]
    colors_baseline = ["#D29393", "#A481B9", "#005B67", "#F8CBA3", "#4B6928"]

    legend = []

    for bt, top_dir in enumerate([top_dir_tuned, top_dir_baseline]):
        path_dict = {}
        df_dict = {}

        test_path_dict = {}

        directories = sorted([top_dir + "/" + d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))])
        algo = list(map(lambda x: str(x)[len(top_dir) + 1:].split("_")[0], directories))

        value_list = []
        for i, algorithm_path in enumerate(directories):
            path_dict[algo[i]] = [algorithm_path + "/" + d + "/progress.csv" for d in os.listdir(algorithm_path) if os.path.isdir(os.path.join(algorithm_path, d))]
            test_path_dict[algo[i]] = [algorithm_path + "/" + d + "/result.json" for d in os.listdir(algorithm_path) if os.path.isdir(os.path.join(algorithm_path, d))]

        for j, (algorithm, paths) in enumerate(test_path_dict.items()):
            if str(algorithm) in ["DDPG", "TD3", "SAC"] if policy == "on" else ["A2C", "PPO"]:
                continue
            mean_list_erm = []
            for p in paths:
                with open(p, 'r') as file:
                    data = json.loads(file.readlines()[-1].strip())
                    erm = data["evaluation"]["sampler_results"]["episode_reward_mean"]
                    mean_list_erm.append(erm)
            print(f"{algorithm}_{env}_episode_reward_mean_mean:  ", mean(mean_list_erm), f"{algorithm}_{env}_episode_reward_mean_std:  ", stdev(mean_list_erm))
            print("------------------------------------------------------------")
            print()

        for j, (algorithm, paths) in enumerate(path_dict.items()):
            if policy == "on":
                if str(algorithm) in ["DDPG", "TD3", "SAC"]:
                    continue  # pass

                df_list = [pandas.read_csv(path) for path in paths]
                df_list = list(map(lambda df: df[["episode_reward_mean", "num_agent_steps_sampled"]], df_list))

                episode_reward_mean = pandas.concat(df_list, axis=1)["episode_reward_mean"].std(axis=1)

                df: pandas.DataFrame = sum(df_list) / len(df_list)
                df["index"] = [x for x in range(len(df.values))]

                # plt.plot(df["num_agent_steps_sampled"], df[metric].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-', color=colors[j])
                plt.plot(df["index"], df["episode_reward_mean"].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-',
                         color=colors_tuned[j] if not bt else colors_baseline[j])
                if "baseline" not in top_dir:
                    legend.append(f"Tuned_{algorithm}")
                else:
                    legend.append(f"Baseline_{algorithm}")
            elif policy == "off":
                if str(algorithm) in ["PPO", "A2C"]:
                    continue  # pass

                df_list = [pandas.read_csv(path) for path in paths]
                df_list = list(map(lambda df: df[["episode_reward_mean", "num_agent_steps_sampled"]], df_list))

                episode_reward_mean = pandas.concat(df_list, axis=1)["episode_reward_mean"].std(axis=1)

                df: pandas.DataFrame = sum(df_list) / len(df_list)
                df["index"] = [x for x in range(len(df.values))]

                # plt.plot(df["num_agent_steps_sampled"], df[metric].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-', color=colors[j])
                plt.plot(df["index"], df["episode_reward_mean"].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-',
                         color=colors_tuned[j] if not bt else colors_baseline[j])
                if "baseline" not in top_dir:
                    legend.append(f"Tuned_{algorithm}")
                else:
                    legend.append(f"Baseline_{algorithm}")
            else:
                sys.exit(1)

    ax.grid()
    ax.legend(legend, loc=4)
    plt.savefig(f"./{env}_episode_reward_mean_{policy}_bt.png", bbox_inches='tight', format='png')
