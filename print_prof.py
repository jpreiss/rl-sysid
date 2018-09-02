import pstats
import sys

name = sys.argv[1]
sort = sys.argv[2]
p = pstats.Stats(name)
p.strip_dirs().sort_stats(sort).print_stats(100)
