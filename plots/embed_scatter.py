import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plots.lib import *
import itertools

def interleave(a, b):
	return itertools.chain.from_iterable(zip(a, b))

# grid stuff from https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
# nested labeling stuff from https://stackoverflow.com/questions/20365122/how-to-make-a-grouped-boxplot-graph-in-matplotlib

# inputs shaep (n_plots 1) (TODO: handle multi-dimensional?)
def embed_scatter(actual, estimated, titles, xlabels):

	n_plots, _ = actual.shape
	fig, axs = plt.subplots(1, n_plots, figsize=(6.5,3.0))
	plt.subplots_adjust(wspace=0.5)

	for ax, ac, est, title, xlab in zip(axs, actual, estimated, titles, xlabels):
		data_range = max(ac) - min(ac)
		bound = int(0.70 * data_range + 0.5)
		span = (-1.2*bound, 1.2*bound)

		ax.scatter(ac, est, c='k', s=5.0, edgecolors='none')
		ax.plot(span, span, 'k', linestyle=":", linewidth=0.75)
		ax.axis('equal')

		ax.set_adjustable("box")
		ax.set_xbound(span)
		ax.set_ybound(span)

		ax.set_xlabel(label2tex(xlab.replace(" ", "\\ ")))
		ax.set_ylabel(label2tex("estimated"), labelpad=3)
		ax.set_title(label2tex(title))

		tix = list(range(-bound, bound + 1))
		ax.set_xticks(tix)
		ax.set_yticks(tix)
		ax.set_xticklabels(ticks2tex(ax.get_xticks()))
		ax.set_yticklabels(ticks2tex(ax.get_yticks()))

		# TODO why doesn't this work??
		#ax.ticklabel_format(useMathText=True)

	#fig.tight_layout()
	return fig


if __name__ == "__main__":
	N = 2
	k = 1000
	actual = np.random.normal(size=(N, k))
	estimated = actual + 0.2 * np.random.normal(size=(N, k))
	fig = embed_scatter(actual, estimated, ["test1", "test2"])
	fig.savefig("test.pdf")
