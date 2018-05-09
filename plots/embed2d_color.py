import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from plots.lib import *
import itertools

def interleave(a, b):
	return itertools.chain.from_iterable(zip(a, b))

# grid stuff from https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
# nested labeling stuff from https://stackoverflow.com/questions/20365122/how-to-make-a-grouped-boxplot-graph-in-matplotlib

def normalize_channels(img):
	img = 0 + img
	for i in range(img.shape[2]):
		chan = img[:,:,i]
		min = np.min(chan.flat)
		max = np.max(chan.flat)
		if max == min:
			img[:,:,i] = 0
		else:
			chan -= min
			chan /= (max - min)
			img[:,:,i] = chan
	return img

# inputs shape (N, 2)
def embed2d_color(img, xrange, yrange, titles):
	fig = plt.figure(figsize=(8,3.5))

	colors = np.array([[0.9, 0.55, 0.2], [0.2, 0.6, 0.9]])
	img = np.flip(img, axis=0)
	img = normalize_channels(img)
	img3 = np.zeros(list(img.shape[:2]) + [3])
	for i in range(2):
		img3 += img[:,:,i][:,:,None] * colors[i][None,None,:]
	lighten = 0.1
	img3 = (1 - lighten) * img3 + lighten
	#plt.hist(img[:,:,0].flat)
	#plt.show()
	#plt.hist(img[:,:,1].flat)
	#plt.show()

	ax = plt.gca()
	legend = [Patch(facecolor=c, edgecolor="black") for c in colors]
	labels = ["$\mathrm{{embed}}[{}]$".format(i) for i in range(2)]
	ax.legend(legend, labels, bbox_to_anchor=(1.52, 1.0),
		frameon=False, fontsize="medium", handlelength=1.0)

	plt.imshow(img3, extent=list(xrange) + list(yrange))
	ax.set_xlabel(titles[0])
	ax.set_ylabel(titles[1])
	ax2tex(ax)
	fig.tight_layout()
	return fig

if __name__ == "__main__":
	k = 100
	t = np.linspace(0, 1, k)
	id1, id2 = np.meshgrid(t, t)
	id = np.concatenate([id1[:,:,None], id2[:,:,None]], axis=2)
	fig = embed2d_color(id, (0,1), (0,1), ["id0", "id1"])
	fig.savefig("test.pdf")
