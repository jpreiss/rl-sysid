
def ticks2tex(ticks):
	return ["${}$".format(t) for t in ticks]

def label2tex(label):
	return "$\\mathrm{{{}}}$".format(label)
