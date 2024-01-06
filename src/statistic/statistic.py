import os
import json
import sys

import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metric = sys.argv[3]  # episode_reward_mean | custom_metrics/valids_mean

    env = sys.argv[2]  # qmarket | ecodispatch

    attribute = [*sys.argv[4:]]

    top_dir = sys.argv[1]

    algorithm = sys.argv[1].split('/')[6]

    clip_low_range = 200
    clip_upper_range = 599
    rolling = 5
    reporting_range = 40

    plt.figure(figsize=(10, 7.5))

    plt.rcParams.update({'font.size': 13})

    plt.xlabel('Training iterations')
    if metric == "episode_reward_mean":
        plt.ylabel('Episode Reward Mean')
    else:
        plt.ylabel('Valid solutions mean')
    plt.grid(True)

    colors = ["blue", "red", "cyan", "green", "purple", "orange", "magenta", "yellowgreen", "dodgerblue", "black"]

    legend = []

    directories = [top_dir + "/" + d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]
    value_list = []
    for i in directories:
        with open(i + "/params.json", 'r') as file:
            j = json.load(file)
            if isinstance(j[attribute[0]], dict):
                if isinstance(j[attribute[0]][attribute[1]], list):
                    value_list.append("_".join([str(o) for o in j[attribute[0]][attribute[1]]]))
                    continue
                value_list.append(j[attribute[0]][attribute[1]])
            elif isinstance(j[attribute[0]], list):
                value_list.append("_".join([str(o) for o in j[attribute[0]]]))
            else:
                value_list.append(j[attribute[0]])
    value_list = sorted(list(dict.fromkeys(value_list)))

    path_dict = {i: [] for i in value_list}

    for i in directories:
        with open(i + "/params.json", 'r') as file:
            j = json.load(file)
        if isinstance(j[attribute[0]], dict):
            if isinstance(j[attribute[0]][attribute[1]], list):
                path_dict["_".join([str(o) for o in j[attribute[0]][attribute[1]]])].append(i + "/progress.csv")
                continue
            path_dict[j[attribute[0]][attribute[1]]].append(i + "/progress.csv")
        elif isinstance(j[attribute[0]], list):
            path_dict["_".join([str(o) for o in j[attribute[0]]])].append(i + "/progress.csv")
        else:
            path_dict[j[attribute[0]]].append(i + "/progress.csv")

    for j, (l, o) in enumerate(path_dict.items()):
        df_list = [pandas.read_csv(path) for path in o]

        df_list = list(map(lambda df: df[["episode_reward_mean", "custom_metrics/valids_mean"]], df_list))

        episode_reward_mean = pandas.concat(df_list, axis=1)[metric].std(axis=1)
        if j == 0:
            print()
        print(f"--------------- {attribute[-1]} {l} ---------------")
        print()

        df: pandas.DataFrame = sum(df_list) / len(df_list)
        df["index"] = [x for x in range(len(df.values))]

        plt.plot(df['index'][clip_low_range:clip_upper_range], df[metric].rolling(window=rolling).mean()[clip_low_range:clip_upper_range], linestyle='-', color=colors[j])

        legend.append(f"{attribute[-1]}={round(l, 5) if not isinstance(l, str) else l}")
        # legend.append(f"critic_hiddens={round(l, 5) if not isinstance(l, str) else l}")

        x = []

        y = []

        err = []

        for t in range(clip_low_range, clip_upper_range, reporting_range):
            if (t + j * (reporting_range // len(path_dict))) > clip_upper_range:
                continue

            x.append(df['index'][t + j * (reporting_range // len(path_dict))])

            y.append(df[metric].rolling(window=rolling).mean()[t + j * (reporting_range // len(path_dict))])

            err.append(episode_reward_mean.rolling(window=rolling).mean()[t + j * (reporting_range // len(path_dict))])

        plt.errorbar(x, y, err, ecolor=colors[j], fmt='', linestyle='')

    plt.legend(legend, fontsize="15", loc=4)
    plt.savefig(f"./{algorithm}_{env}_{metric.split('/')[-1]}_{attribute[-1]}.png", bbox_inches='tight', format='png')  # plt.show()
