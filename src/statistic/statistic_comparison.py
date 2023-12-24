import os
import json
import sys
import numpy
from statistics import mean, stdev

import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metric = sys.argv[1]  # episode_reward_mean | custom_metrics/valids_mean

    env = sys.argv[2]  # qmarket | ecodispatch

    top_dir = f"/fs/dss/work/zahl4814/comparison_long/{env}"

    clip_upper_range = 3_000_000

    fig = plt.figure(figsize=(10, 7.5))
    plt.xlabel('Training iterations')
    plt.ylabel('Episode reward mean')

    # plt.xlabel('Steps sampled')

    if env == "qmarket":
        plt.ylim(-0.15, 0.0)
    else:
        plt.ylim(-1.5, -0.5)

    ax = fig.gca()

    if env == 'qmarket':
        ax.set_yticks(numpy.arange(-0.15, 0.0, 0.01))
    else:
        ax.set_yticks(numpy.arange(-1.5, -0.5, 0.1))

    colors = ["blue", "red", "cyan", "green", "purple", "orange", "magenta", "yellowgreen", "dodgerblue", "black"]

    legend = []
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
        mean_list_erm = []
        mean_list_vsm = []
        for p in paths:
            with open(p, 'r') as file:
                data = json.loads(file.readlines()[-1].strip())
                erm, vsm = data["evaluation"]["sampler_results"]["episode_reward_mean"], data["evaluation"]["sampler_results"]["custom_metrics"]["valids_mean"]
                mean_list_erm.append(erm)
                mean_list_vsm.append(vsm)
        print(f"{algorithm}_{env}_episode_reward_mean_mean:  ", mean(mean_list_erm), f"{algorithm}_{env}_episode_reward_mean_std:  ", stdev(mean_list_erm))
        print("------------------------------------------------------------")
        print()

    for j, (algorithm, paths) in enumerate(path_dict.items()):
        if str(algorithm) not in ["DDPG", "TD3", "SAC"]:
            continue  # pass

        df_list = [pandas.read_csv(path) for path in paths]
        df_list = list(map(lambda df: df[["episode_reward_mean", "custom_metrics/valids_mean", "num_agent_steps_sampled"]], df_list))

        episode_reward_mean = pandas.concat(df_list, axis=1)[metric].std(axis=1)

        df: pandas.DataFrame = sum(df_list) / len(df_list)
        df["index"] = [x for x in range(len(df.values))]

        # plt.plot(df["num_agent_steps_sampled"], df[metric].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-', color=colors[j])
        plt.plot(df["index"], df[metric].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-', color=colors[j])

        legend.append(f"{algorithm}")

        x = []

        y = []

        err = []

    ax.grid()
    ax.legend(legend, loc=4)
    plt.savefig(f"./{env}_{metric.split('/')[-1]}.png", bbox_inches='tight', format='png')
