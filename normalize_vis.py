import collections
import random

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

VISUALIZE_DATASET = True
START_OFFSET = 0


def main():
    random.seed(229)

    infile = "data.csv"
    data = pd.read_csv(infile)

    epoch = data["epoch"].values
    names = list(data)[1:]

    smoothing = 1
    x = epoch[START_OFFSET:-smoothing-1]

    colors = iter(plt.cm.rainbow(np.linspace(0,1,30)))

    if VISUALIZE_DATASET:
        # for name in names:
        #     seq = data[name].values[START_OFFSET:-2]
        #
        #     # seq -= np.mean(seq)
        #     seq -= seq[0]
        #     seq /= np.max(seq) - np.min(seq)
        #
        #
        #     # plt.scatter(x, seq, color=next(colors), marker='x', alpha=0.2, linewidths=0.5)
        #     plt.plot(x, seq, color=next(colors), alpha=0.5)

        seq = data[names[0]].values[START_OFFSET:-2]

        y1 = seq[:-1]
        y2 = seq[1:]
        diff = y2 - y1

        plt.scatter(x[1:], diff, color="blue", marker='.', alpha=0.2, linewidths=0.7)
        plt.axhline(y=0, linestyle="--", color="grey")

        plt.xlim([0, 150])

        plt.xticks(np.arange(0, 150.1, 10))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Difference between Epochs")
        plt.savefig("acc_diff.png")
        # plt.show()
        return


if __name__=='__main__':
    main()
