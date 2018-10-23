import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


# rews shapes: [iter, run] (runs could be per-seed or per-seed-agent)
# style should be one of:
#   "each"  - plot all seeds as individual lines
#   "std"   - plot mean and one-std bounds over seeds
#   "bound" - plot 5% / 95% reward percentiles
# boxwidth: width of box moving average filter
def learning_curves(labels: List[str], rews: List[np.ndarray],
    style: str="std", boxwidth: int=11):

    ntypes = len(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, ntypes))

    def moving_average(x):
        kernel = 1.0 / boxwidth * np.ones(boxwidth)
        return np.convolve(x, kernel, mode="valid")

    for label, color, r in zip(labels, colors, rews):
        if style == "bound":
            mins = np.percentile(r, 5, axis=-1)
            maxes = np.percentile(r, 95, axis=-1)
            mins = moving_average(mins)
            maxes = moving_average(maxes)
            x = np.arange(len(mins)) + boxwidth // 2 + 1
            plt.fill_between(x, mins, maxes, color=color, alpha=0.1)
            plt.plot(x, mins, color=color, linewidth=1.0, label=label)
            plt.plot(x, maxes, color=color, linewidth=1.0)
        elif style == "std":
            means = moving_average(np.mean(r, axis=-1))
            stds = moving_average(np.std(r, axis=-1))
            x = np.arange(len(means)) + boxwidth // 2 + 1
            plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.1)
            plt.plot(x, means, color=color, linewidth=2.0, label=label)
        elif style == "each":
            handles = []
            for run in r.T:
                handles += plt.plot(run, color=color)
            handles[0].set_label(label)
        else:
            raise ValueError("invalid plot style")


    plt.grid(True, axis="y")
    plt.xlabel('iteration')
    plt.ylabel('reward per episode')
    plt.legend()
    return plt.gcf()

if __name__ == "__main__":
    horizon = 50
    N = 10
    x = np.linspace(0, 1, horizon)
    mean = x[:,None] ** 2
    rews = mean + (0.2 + 0.5*mean) * np.random.normal(size=(len(x), N))
    # we generated (iter, N)
    # routine expects (flavor, alpha, seed, iter, N)
    rews = rews[None,None,None,:,:]
    modes = ("each", "std", "bound")
    for i, mode in enumerate(modes):
        plt.subplot(len(modes), 1, i + 1)
        learning_curves(["flav_test"], ["0.1"], rews, mode)
    plt.show()
