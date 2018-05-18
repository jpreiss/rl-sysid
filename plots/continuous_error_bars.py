import numpy as np
import matplotlib.pyplot as plt

def shaded_errors(x, y, ystd, color):
    plt.plot(x, y, color=color, linewidth=2.0)
    plt.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.2)

