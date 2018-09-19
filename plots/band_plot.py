import numpy as np
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
def band(x, ys, mode, label="", color=None, smoothing=1, percentile=5.0, sigmas=1.0):

    def moving_average(x):
        kernel = 1.0 / smoothing * np.ones(smoothing)
        return np.convolve(x, kernel, mode="valid")

    if mode == "bound":
        mins = moving_average(np.percentile(ys, percentile, axis=-1))
        maxes = moving_average(np.percentile(ys, 100.0 - percentile, axis=-1))
        ind = np.arange(len(mins)) + smoothing // 2 + 1
        xind = np.array(x)[ind]
        line = plt.plot(xind, mins, color=color, linewidth=1.0, label=label)[0]
        if color is None:
            color = line.get_color()
        plt.plot(xind, maxes, color=color, linewidth=1.0)
        plt.fill_between(xind, mins, maxes, color=color, alpha=0.1)
    elif mode == "std":
        means = moving_average(np.mean(ys, axis=-1))
        stds = moving_average(np.std(ys, axis=-1))
        x = np.arange(len(means)) + smoothing // 2 + 1
        line = plt.plot(x, means, color=color, linewidth=2.0, label=label)[0]
        if color is None:
            color = line.get_color()
        plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.1)
    else:
        for run in ys.T:
            plt.plot(run, color=color)
