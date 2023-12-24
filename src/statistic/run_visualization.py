import json

import numpy
import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "/fs/dss/work/zahl4814/comparison_long/qmarket/DDPG_qmarket_comparison/DDPG_QMarketEnv-v0_45b26_00000_0_2023-12-14_19-40-36/"
    algorithm = "DDPG"
    env = "qmarket"

    fig = plt.figure(figsize=(10, 7.5))
    plt.xlabel('Training iterations')
    plt.ylabel('Episode reward mean')

    if env == "qmarket":
        plt.ylim(-0.15, 0.0)
    else:
        plt.ylim(-1.5, -0.5)

    ax = fig.gca()

    if env == 'qmarket':
        ax.set_yticks(numpy.arange(-0.15, 0.0, 0.01))
    else:
        ax.set_yticks(numpy.arange(-1.5, -0.5, 0.1))

    with open(path + "result.json", 'r') as file:
        data = json.loads(file.readlines()[-1].strip())
        erm = data["evaluation"]["sampler_results"]["episode_reward_mean"]
    print(f"{algorithm}_{env}_episode_reward_mean_mean:  ", erm)

    df = pandas.read_csv(path + "progress.csv")
    df["index"] = [x for x in range(len(df.values))]




    plt.plot(df["index"], df["episode_reward_mean"].rolling(window=500 if algorithm in ["DDPG", "TD3", "SAC"] else 5).mean(), linestyle='-',color="purple")

    ax.grid()
    ax.legend([f"{algorithm}"], loc=4)

    plt.savefig(f"./{env}_{algorithm}_visualization.png", bbox_inches='tight', format='png')