import matplotlib.pyplot as plt

# plot a "band of lines" in multiple ways
# x: [N]
# ys: [N, M]
# type: case of
#   "bound" : shades area btw (pct) and (1 - pct) percentiles.
#   "std"   : plots heavy line for mean w/ shaded region +/- sigmas
#   "lines" : plots a different line for each of the M lines
# smoothing: width of moving average filter
# percentile: used in "bound" mode, default = 5% (implies upper limit 95%)
# sigmas: used in "std" mode, default = 1 (one standard deviation)
def band(x, ys, color, smoothing=1, percentile=5.0, sigmas=1.0):

    def moving_average(x):
        kernel = 1.0 / smoothing * np.ones(smoothing)
        return np.convolve(x, kernel, mode="valid")

    if mode == "bound":
        mins = moving_average(np.percentile(r, percentile, axis=-1))
        maxes = moving_average(np.percentile(r, percentile, axis=-1))
        ind = np.arange(len(mins)) + boxwidth // 2 + 1
        plt.fill_between(x, mins, maxes, color=color, alpha=0.1)
        plt.plot(x, mins, color=color, linewidth=1.0, label=label)
        plt.plot(x, maxes, color=color, linewidth=1.0)
    elif mode == "std":
        means = moving_average(np.mean(r, axis=-1))
        stds = moving_average(np.std(r, axis=-1))
        x = np.arange(len(means)) + boxwidth // 2 + 1
        plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.1)
        plt.plot(x, means, color=color, linewidth=2.0, label=label)
    else:
        for run in r.T:
            plt.plot(run, color=color)
