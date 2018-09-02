import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import itertools

# grid stuff from https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
# nested labeling stuff from https://stackoverflow.com/questions/20365122/how-to-make-a-grouped-boxplot-graph-in-matplotlib

# rews shape: (n_flavors, n_alphas, n_runs)
def reward_boxplot(flavors, alphas, train_rews, test_rews):

	fig = plt.figure(figsize=(8,4.0))
	outer = gridspec.GridSpec(1, len(flavors), wspace=0, top=0.8, bottom=0.2)

	ysrc = None
	ylim = None

	whitebox = None
	blackbox = None

	import pdb
	pdb.set_trace()

	all_axes = []
	ylim = [float("inf"), -float("inf")]

	for i, flavor in enumerate(flavors):

		ax = plt.Subplot(fig, outer[i])
		for spine in ax.spines.values():
			spine.set_visible(False)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel("$\\mathrm{{{}}}$".format(flavor), labelpad=25.0)
		#ax.set_aspect(8.0)
		fig.add_subplot(ax)

		inner = gridspec.GridSpecFromSubplotSpec(1, len(alphas),
			subplot_spec=outer[i], wspace=0)

		for j, alpha in enumerate(alphas):
			ax = plt.Subplot(fig, inner[j])

			ax.grid(True, axis="y", color="gray",
				linewidth=0.75, dashes = (1.0, 3.0))
			ax.set_axisbelow(True) # move the grid lines behind the box plots

			p = ax.boxplot([train_rews[i,j,:], test_rews[i,j,:]], 
				sym="",
				whis=1.0,
				patch_artist=True, widths=0.5)

			# set the box plot's style
			p["boxes"][0].set_facecolor("white")
			p["boxes"][1].set_facecolor("black")
			whitebox = p["boxes"][0]
			blackbox = p["boxes"][1]
			p["medians"][1].set_color("white")
			for w in p["whiskers"]:
				w.set_linestyle("-")
				w.set_color("black")
				w.set_linewidth(1.5)
			for c in p["caps"]:
				c.set_linestyle("None")

			# set the axes style
			plt.setp(ax.get_xticklabels(), visible=False)
			for spine in ax.spines.values():
				spine.set_visible(False)
			ax.tick_params(axis="y", which="both", left=False, right=False)
			ax.tick_params(axis="x", which="both", top=False, bottom=False)

			# add padding space that still has the grid lines in it
			ax.set_xlim([0.0, 3.0])

			# align y axes, small expansion makes sure grid lines aren't clipped
			# TODO: handle case where ylim from first isn't valid for rest
			y0, y1 = ax.get_ylim()
			ylim[0] = min(y0, ylim[0])
			ylim[1] = max(y1, ylim[1])

			# show y ticks only on the left-most plot
			if ysrc is not None:
				plt.setp(ax.get_yticklabels(), visible=False)
				fig.add_subplot(ax, sharey=ysrc)
			else:
				fig.add_subplot(ax)
				ysrc = ax

			ax.set_xlabel("$\\alpha = {}$".format(alpha))
			all_axes.append(ax)
	
	expand = 0.01 * (ylim[1] - ylim[0])
	ylim = [ylim[0] - expand, ylim[1] + expand]
	for ax in all_axes:
		ax.set_ylim(ylim)

	tix = ysrc.get_yticks()
	# force to use LaTeX-style interpreter
	tix = ["${}$".format(int(t)) for t in tix]
	ysrc.set_yticklabels(tix)

	plt.legend((whitebox, blackbox), (r"$\mathrm{train}$", r"$\mathrm{test}$"), 
		ncol=2, bbox_to_anchor=(1.05, 1.25), frameon=False, fontsize="medium",
		handlelength=1.0,
	)
	return fig


if __name__ == "__main__":
	flavors = ["blind", "plain", "extra", "embed"]
	alphas = [0.0, 0.1]
	train_rews = np.random.uniform(size=(4, 2, 100))
	test_rews = np.random.uniform(size=(4, 2, 50))
	fig = reward_boxplot(flavors, alphas, train_rews, test_rews)
	fig.savefig("test.pdf")
