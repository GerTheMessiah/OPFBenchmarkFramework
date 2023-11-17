import os
import json
import sys

import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metric = sys.argv[3]  # episode_reward_mean | custom_metrics/valids_mean

    cut = int(sys.argv[4])

    value_ = float(sys.argv[5])

    env = sys.argv[2]  # qmarket | ecodispatch

    attribute = [*sys.argv[6:]]

    top_dir = sys.argv[1]

    algorithm = sys.argv[1].split('/')[5]

    # top_dir = "/user/zahl4814/ray_results/A2C_2023-10-13_01-30-18"

    # actor_lr | actor_hiddens | critic_lr | critic_hiddens | train_batch_size | exploration_config - stddev | tau | entropy_learning_rate | entropy_coeff | grad_clip | model - fcnet_hiddens | sgd_minibatch_size | vf_loss_coeff | num_sgd_iter

    plt.figure(figsize=(10, 7.5))
    # plt.title(f'DDPG Training Data - Valid Solutions - {attribute[-1]}')
    plt.xlabel('Training iterations')
    plt.ylabel(metric.split('/')[-1].replace("_", " ").capitalize())
    plt.grid(True)
    colors = ["blue", "red", "cyan", "green", "purple", "orange", "magenta", "yellowgreen", "dodgerblue", "black"]

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

    # for i, j in path_dict.items():
    #     print(f"Value: {i}")
    #     for k in j:
    #         print(k)


    df_list = [pandas.read_csv(path) for path in path_dict.get(value_)]
    df_list = list(map(lambda df: df[["episode_reward_mean", "custom_metrics/valids_mean"]], df_list))


    for j, df in enumerate(df_list):
        df["index"] = [x for x in range(len(df.values))]
        plt.plot(df["index"][cut:], df[metric].rolling(window=400).mean()[cut:], linestyle='-', color=colors[j])

    plt.legend([f"{attribute[-1]}={i}" for i in range(len(df_list))], loc=4)
    # plt.savefig(f"./{algorithm}_{env}_{metric.split('/')[-1]}_{attribute[-1]}.png", bbox_inches='tight', format='png')
    plt.show()