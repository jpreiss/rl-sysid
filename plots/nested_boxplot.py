import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import itertools

# grid stuff from https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot
# nested labeling stuff from https://stackoverflow.com/questions/20365122/how-to-make-a-grouped-boxplot-graph-in-matplotlib

def nested_boxplot(label_sets, *box_datas, aspect_ratio=2.0):

	dims = [len(labels) for labels in label_sets]
	n_outer, n_inner, n_boxes = dims
	assert len(box_datas) == n_boxes

	# determine figure size
	width = np.product(dims) ** 0.75
	height = width / aspect_ratio
	fig = plt.figure(figsize=(width, height))

	# determine colors
	cmap = plt.get_cmap("tab10")
	assert n_boxes <= cmap.N
	colors = np.array([cmap(i) for i in range(n_boxes)])
	print(colors.shape)

	# set up outermost level
	outer = gridspec.GridSpec(1, dims[0], wspace=0, top=0.8, bottom=0.2)
	ax_with_yticks = None
	all_axes = []
	ylim = [float("inf"), -float("inf")]

	for i, outlab in enumerate(label_sets[0]):
		ax = plt.Subplot(fig, outer[i])
		for spine in ax.spines.values():
			spine.set_visible(False)
		ax.set_xticks([])
		ax.set_yticks([])

		ax.set_xlabel(f"$\\mathrm{{{outlab}}}$", labelpad=25.0)
		
		#ax.set_aspect(8.0)
		fig.add_subplot(ax)

		inner = gridspec.GridSpecFromSubplotSpec(1, dims[1],
			subplot_spec=outer[i], wspace=0)

		for j, inlab in enumerate(label_sets[1]):
			ax = plt.Subplot(fig, inner[j])

			# magic numbers specify the gridline dash style
			ax.grid(True, axis="y",
				color="gray", linewidth=0.75, dashes = (1.0, 3.0))
			ax.set_axisbelow(True)  # move the grid lines behind the box plots

			whisker_percentile=1.0
			p = ax.boxplot([d[i][j] for d in box_datas],
				sym="",
				whis=whisker_percentile,
				patch_artist=True, widths=0.5)
			boxes = p["boxes"]

			# TODO can we move some of these retroactive changes to the boxplot args?

			# set the colors.
			for box, median, color in zip(boxes, p["medians"], colors):
				box.set_facecolor(color)
				weights = [0.241, 0.691, 0.068]
				brightness = np.sqrt(np.sum(weights * color[:-1]))
				if brightness < 0.7:
					median.set_color("white")
				else:
					median.set_color("black")

			# set the whisker styles.
			for w in p["whiskers"]:
				w.set_linestyle("-")
				w.set_color("black")
				w.set_linewidth(1.5)
			for c in p["caps"]:
				c.set_linestyle("None")

			# set the axes style.
			plt.setp(ax.get_xticklabels(), visible=False)
			for spine in ax.spines.values():
				spine.set_visible(False)
			ax.tick_params(axis="y", which="both", left=False, right=False)
			ax.tick_params(axis="x", which="both", top=False, bottom=False)

			# expand xlim instead of using padding so the grid is not interrupted
			ax.set_xlim([0.0, dims[2] + 1.0])

			# keep running union of ylims
			y0, y1 = ax.get_ylim()
			ylim[0] = min(y0, ylim[0])
			ylim[1] = max(y1, ylim[1])

			# show y ticks only on the left-most (and right-most, later) plots
			if ax_with_yticks is not None:
				plt.setp(ax.get_yticklabels(), visible=False)
				fig.add_subplot(ax, sharey=ax_with_yticks)
			else:
				fig.add_subplot(ax)
				ax_with_yticks = ax

			ax.set_xlabel(inlab)
			all_axes.append(ax)

	# expand to make sure we get the outer grid lines
	# TODO does this mean we can get rid of sharey arg to add_subplot above?
	expand = 0.01 * (ylim[1] - ylim[0])
	ylim = [ylim[0] - expand, ylim[1] + expand]
	for ax in all_axes:
		ax.set_ylim(ylim)

	# force to use LaTeX-style interpreter
	# TODO allow not-integer labels
	tix = [f"${int(t)}$" for t in ax_with_yticks.get_yticks()]
	ax_with_yticks.set_yticklabels(tix)

	# right side y-ticks
	right_ax = all_axes[-1]
	right_ax.yaxis.tick_right()
	plt.setp(right_ax.get_yticklabels(), visible=True)
	right_ax.set_yticklabels(tix)
	right_ax.tick_params(axis="y", which="both", left=False, right=False)

	plt.legend(boxes, (f"$\\mathrm{{{boxlab}}}$" for boxlab in label_sets[2]),
		ncol=dims[2], bbox_to_anchor=(1.05, 1.25), frameon=False, fontsize="medium",
		handlelength=1.0,
	)
	return fig


if __name__ == "__main__":
	labels = [
		["blind", "plain", "extra", "embed"],
		[0.0, 0.1],
		["train", "test"],
	]
	sizes = np.random.randint(10, 100, size=len(labels[2]))
	datasize = (len(labels[0]), len(labels[1]))
	datas = [np.random.normal(size=datasize + (sz,)) for sz in sizes]

	fig = nested_boxplot(labels, *datas)
	fig.savefig("test.pdf")
