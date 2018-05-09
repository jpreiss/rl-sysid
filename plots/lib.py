import numpy as np
import itertools as it

def ticks2tex(ticks, decimals=None):
	def find_decimals(ticks):
		for decimals in it.count(0):
			strs = set("${:.{}f}$".format(t, decimals) for t in ticks)
			if len(strs) == len(ticks):
				return decimals

	if decimals is None:
		decimals = find_decimals(ticks)

	return ["${:.{}f}$".format(t, decimals) for t in ticks]

def label2tex(label):
	return "$\\mathrm{{{}}}$".format(label)

def ax2tex(ax):
	ax.set_xticklabels(ticks2tex(ax.get_xticks()))
	ax.set_yticklabels(ticks2tex(ax.get_yticks()))
	ax.set_xlabel(label2tex(ax.get_xlabel()))
	ax.set_ylabel(label2tex(ax.get_ylabel()))
	ax.set_title(label2tex(ax.get_title()))
