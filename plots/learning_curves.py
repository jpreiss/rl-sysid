import itertools as it
import numpy as np
import matplotlib.pyplot as plt


# mode should be one of:
#   "each" - plot all seeds as individual lines
#   "std"  - plot mean and one-std bounds over seeds
def learning_curves(flavors, alphas, per_seed_rews, mode="std"):
    plt.clf() # TODO specify figure size:
    plt.hold(True)
    color_list = plt.cm.Set3(np.linspace(0, 1, len(list(it.product(flavors)))))

    # smooth individual runs
    def smooth_all(rews):
        box_width = 7
        kernel = 1.0 / box_width * np.ones(box_width)
        return [np.convolve(seedrew, kernel, mode="valid") for seedrew in rews]

    for flavor, alpha, rews, color in zip(flavors, alphas, per_seed_rews, color_list):
        smoothed = smooth_all(rews)
        if mode == "each":
            for seed_rew in smoothed:
                plt.plot(seed_rew, color=color, linewidth=2,
                    label=str((flavor, alpha)))
        elif mode == "std":
            means = np.mean(smoothed, axis=0)
            stds = np.std(smoothed, axis=0)
            x = np.arange(len(means))
            plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
            plt.plot(x, means, color=color, linewidth=2.0, label=str((flavor, alpha)))

    plt.grid(True, axis="y")
    plt.xlabel('iteration')
    plt.ylabel('reward per episode')
    plt.legend(loc='lower right')
    return plt.gcf()


if __name__ == "__main__":
    N = 50
    x = np.linspace(0, 1, N)
    mean = x ** 2
    rews = mean[None,:] + (0.2 + 0.5*mean)[None,:] * np.random.normal(size=(3, len(x)))
    plt.clf()
    plt.subplot(1, 2, 1)
    learning_curves(["flav"], ["0.1"], [rews], "each")
    plt.subplot(1, 2, 2)
    learning_curves(["flav"], ["0.1"], [rews], "std")
    plt.show()
